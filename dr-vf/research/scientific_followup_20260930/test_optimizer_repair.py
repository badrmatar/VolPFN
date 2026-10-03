import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from .optimizer_core import ROOT, fit, FitFailure, legacy, fast_gas_path, gas_objective, starts_for
from .optimizer_paths import specialist_paths
from .optimizer_run import wrapped_pairs
from .optimizer_statistics import all_boots,correct_selection_labels,validated_roster,CANONICAL_MARKETS,analysis


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = legacy.load_market("SPX",ROOT/"research/data/oxfordman_barde/calendar_full_v1")["r"]

    def test_compiled_recurrence_and_objective_match_python(self):
        for alpha in legacy.ALPHAS:
            r = self.r[:1500]
            for th in starts_for(r,"gas_fz0",alpha):
                params = legacy._gas_unpack(th)
                v,e = legacy.gas_path(r,*params,alpha)
                vv,ee = fast_gas_path(r,*params,alpha)
                np.testing.assert_allclose(vv,v,rtol=2e-14,atol=2e-14)
                np.testing.assert_allclose(ee,e,rtol=2e-14,atol=2e-14)
                self.assertAlmostEqual(gas_objective(r,alpha)(th),float(np.mean(legacy.fz0(r[50:],v[50:],e[50:],alpha))),places=13)

    def test_representative_optimizer_parity(self):
        r = self.r[:750]; alpha = .01
        def py_loss(th):
            p = legacy._gas_unpack(th)
            if abs(p[3])>1/alpha:return 1e6
            v,e = legacy.gas_path(r,*p,alpha)
            return float(np.mean(legacy.fz0(r[50:],v[50:],e[50:],alpha)))
        for th in starts_for(r,"gas_fz0",alpha):
            opts = dict(maxfev=400,xatol=1e-4,fatol=1e-8)
            py = minimize(py_loss,th,method="Nelder-Mead",options=opts)
            compiled = minimize(gas_objective(r,alpha),th,method="Nelder-Mead",options=opts)
            self.assertEqual(py.success,compiled.success)
            self.assertEqual(py.nfev,compiled.nfev)
            np.testing.assert_allclose(py.x,compiled.x,rtol=2e-13,atol=2e-13)
            self.assertAlmostEqual(py.fun,compiled.fun,places=13)

    def test_consequential_failure_receives_converged_repair(self):
        th,p,receipt = fit(self.r[:750],"gas_fz0",.01)
        self.assertFalse(receipt["initial_best_converged"])
        self.assertTrue(receipt["selected_converged"])
        self.assertGreater(receipt["objective_improvement"],.05)
        self.assertTrue(receipt["accepted"])

    def test_no_success_is_not_accepted(self):
        def failed(*a,**k):
            return OptimizeResult(x=np.asarray(a[1]),fun=1.,success=False,status=1,message="injected exhausted budget",nit=1,nfev=1)
        with patch("research.scientific_followup_20260930.optimizer_core.minimize",failed):
            with self.assertRaises(FitFailure) as caught:
                fit(self.r[:750],"gas_fz0",.01)
        self.assertFalse(caught.exception.receipt["accepted"])
        self.assertEqual(len(caught.exception.receipt["attempts"]),6)

    def test_future_and_target_perturbation_and_prefix_sharing(self):
        r = self.r[:1300].copy(); changed = r.copy(); changed[1000:] = 7*changed[1000:]+2
        for model,alpha in (("garch_t",None),("gas_fz0",.05),("gas_fz0",.025),("gas_fz0",.01)):
            v,e,rr,failed = specialist_paths(r,model,alpha)
            vv,ee,ss,_ = specialist_paths(changed,model,alpha)
            np.testing.assert_array_equal(v[:1001],vv[:1001]); np.testing.assert_array_equal(e[:1001],ee[:1001])
            self.assertEqual(rr[1]["parameters"],ss[1]["parameters"])
            vp,ep,_,_ = specialist_paths(r[:1100],model,alpha)
            np.testing.assert_array_equal(v[:1100],vp); np.testing.assert_array_equal(e[:1100],ep)

    def test_failed_refit_carries_physical_parameters(self):
        from . import optimizer_paths as paths
        real = paths.fit
        def fail_second(r,*a,**k):
            if len(r)==1000 and k.get("repaired",True):raise FitFailure(dict(accepted=False,reason="injected",attempts=[]))
            return real(r,*a,**k)
        with patch.object(paths,"fit",fail_second):
            v,e,receipts,failed=paths.specialist_paths(self.r[:1100],"gas_fz0",.01)
        self.assertTrue(failed[1000:].all()); self.assertFalse(failed[:1000].any())
        self.assertEqual([receipts[0][k] for k in ("a","b","beta","gamma")],[receipts[1][k] for k in ("a","b","beta","gamma")])
        self.assertTrue(np.isfinite(v).all()); self.assertTrue(np.isfinite(e).all())

    def test_bootstrap_multiplicities_preserve_legacy_missing_calendar_arithmetic(self):
        rng=np.random.default_rng(871);daily={};dates={}
        union=np.datetime64("2011-01-01")+np.arange(203)
        for j in range(5):
            ix=np.sort(rng.choice(203,size=170+j,replace=False));m=str(j)
            dates[m]=union[ix].astype("U10");daily[m]={"a":rng.normal(size=len(ix)),"b":rng.normal(size=len(ix))}
        mat,ms=legacy.diff_matrix(daily,dates,"a","b")
        old=legacy.paired_block(mat);new=all_boots(daily,dates,[("a","b")])[0]
        for key in old:np.testing.assert_allclose(new[key],old[key],rtol=2e-14,atol=2e-14)

    def test_repaired_forecasters_and_all_wrapper_paths_are_causal(self):
        r=self.r[:1300].copy();changed=r.copy();changed[1000:]=3*changed[1000:]+4
        h,nu,glog,_=specialist_paths(r,"garch_t")
        cache=dict(h_garch=h,nu_garch=nu,garch_log=glog)
        checks=0
        for alpha in legacy.ALPHAS:
            for fc in ("garch_gauss","garch_t","fhs_garch","gas_fz0"):
                if fc=="gas_fz0":
                    v,e,_,_=specialist_paths(r,fc,alpha)
                else:v,e=legacy.forecast(fc,r,alpha,cache)
                v=np.minimum(v,legacy.V_CAP);e=np.minimum(e,v-1e-6)
                original=wrapped_pairs(r,v,e,alpha);perturbed=wrapped_pairs(changed,v,e,alpha)
                for cb in original:
                    for old,new in zip(original[cb],perturbed[cb]):np.testing.assert_array_equal(old[:1001],new[:1001])
                    checks+=1
        self.assertEqual(checks,180)

    def test_later_retrospective_choice_never_becomes_development_selection(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for label,choice in (("development","ratio_win1000"),("later","ratio_win2000")):
                root=out/label;root.mkdir()
                (root/"decisions.json").write_text(json.dumps([dict(rule="M4_variant_choice",chosen=choice),dict(rule="M2_no_harm",passed=False,excess=.014327672)]))
                (root/"comparison_to_archive.json").write_text(json.dumps(dict(repaired_development_choice=choice,score=.72318)))
            (out/"summary.json").write_text(json.dumps(dict(status="complete_comparisons")))
            correct_selection_labels(out)
            later=json.loads((out/"later/comparison_to_archive.json").read_text())
            self.assertEqual(later["repaired_development_choice"],"ratio_win1000")
            self.assertEqual(later["retrospective_later_m4_maximizer"],"ratio_win2000")
            self.assertEqual(later["score"],.72318)
            decisions=json.loads((out/"later/decisions.json").read_text())
            self.assertFalse(decisions[1]["passed"]);self.assertEqual(decisions[1]["excess"],.014327672)
            self.assertIn("retrospective later diagnostic",decisions[0]["status"])

    def test_standalone_statistics_reject_partial_and_noncanonical_protocols_before_bootstrap(self):
        from pathlib import Path
        for roster in (["SPX"],list(CANONICAL_MARKETS[:-1])+["UNKNOWN"]):
            with self.subTest(roster=roster),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);(out/"protocol.json").write_text(json.dumps(dict(markets=roster)))
                with patch("research.scientific_followup_20260930.optimizer_statistics.all_boots") as boot:
                    with self.assertRaisesRegex(ValueError,"canonical 28-market"):
                        analysis(out,"development")
                boot.assert_not_called();self.assertFalse((out/"summary.json").exists())

    def test_standalone_statistics_require_passed_complete_quality_receipts(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/"protocol.json").write_text(json.dumps(dict(markets=CANONICAL_MARKETS)))
            with self.assertRaisesRegex(ValueError,"quality receipts"):
                validated_roster(out)
            quality=dict(status="passed_complete_coverage_actual_archival_quality_and_preservation_gates",repaired_nonconverged=0,
                         scored_substitutions=0,cohorts=[dict(cohort=label,market=m) for label in ("development","later") for m in CANONICAL_MARKETS])
            (out/"quality.json").write_text(json.dumps(quality))
            (out/"fhs_optimizer_quality.json").write_text(json.dumps(dict(status="passed_independent_actual_FHS_optimizer_archive_gate")))
            self.assertEqual(validated_roster(out),list(CANONICAL_MARKETS))
            for change in (dict(status="failed"),dict(scored_substitutions=1),dict(cohorts=quality["cohorts"][:-1])):
                with self.subTest(change=change):
                    (out/"quality.json").write_text(json.dumps(dict(quality,**change)))
                    with self.assertRaisesRegex(ValueError,"quality gates"):
                        validated_roster(out)
            (out/"quality.json").write_text(json.dumps(quality))
            (out/"fhs_optimizer_quality.json").write_text(json.dumps(dict(status="failed")))
            with self.assertRaisesRegex(ValueError,"quality gates"):
                validated_roster(out)


if __name__ == "__main__": unittest.main()

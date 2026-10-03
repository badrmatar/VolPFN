"""Validate every actual archived endpoint, coverage, receipts and preservation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .optimizer_core import ROOT,SIBLING,legacy,fast_gas_path
from .optimizer_run import save,sha
from research.audit_support.strict_tail_inputs import load_tail_inputs


def audit_fhs(out):
    out=Path(out);records=[]
    for label,arch,data,extract,split in (("development","market_dev_fhs_v1","calendar_development_v2","attribution_extract_v1",1),
                                        ("later","market_reserved_fhs_v1","calendar_full_v1","attribution_extract_full_v1",2)):
        for cell in sorted((out/label/"cells").glob("*.json")):
            meta=json.loads(cell.read_text());new=json.loads(Path(meta["fits"]["path_file"]).read_text())["garch"]
            idx={f["t"]:f for f in new};oldpath=SIBLING/f"outputs/{arch}/cells/{cell.name}"
            old=json.loads(oldpath.read_text())["fits"]["garch"]
            inputs=load_tail_inputs(ROOT/f"research/data/oxfordman_barde/{data}/{cell.stem}.npz",ROOT/f"research/outputs/{extract}/{cell.stem}.npz",scored_split=split)
            r=inputs.returns_percent
            for f in old:
                t=f["t"];h=legacy.garch_variance(r[:t],f["omega"],f["alpha"],f["beta"],float(np.mean(r[:t]**2)))
                actual=-float(np.mean(legacy.t_loglik(r[1:t],h[1:],f["nu"])))
                if abs(actual-f["nll"])>1e-10:raise ValueError("FHS archived training objective mismatch")
                n=idx[t]
                if not n["accepted"] or not n["selected_converged"] or n["objective"]>actual+1e-7:raise ValueError("Actual FHS archive quality gate failed")
                records.append(dict(cohort=label,market=cell.stem,origin=t,actual_FHS_archived_GARCH_objective=actual,
                                    repaired_objective=n["objective"],repaired_training_improvement=actual-n["objective"],
                                    archived_objective_reproduction_error=actual-f["nll"],archive_path=str(oldpath),archive_sha256=sha(oldpath)))
    result=dict(status="passed_independent_actual_FHS_optimizer_archive_gate",records=records,refit_cohort_records=len(records),
                max_archived_objective_reproduction_error=max(abs(x["archived_objective_reproduction_error"]) for x in records),
                max_repaired_objective_deterioration=max(-x["repaired_training_improvement"] for x in records))
    if (out/"fhs_optimizer_quality.json").exists():raise FileExistsError("Refusing FHS quality overwrite")
    save(out/"fhs_optimizer_quality.json",result)
    return result


def audit(out):
    out=Path(out);records=[];preservation=[];unchanged=[];cohorts=[]
    names=json.loads((out/"protocol.json").read_text())["markets"]
    for label,arch,data,extract,split in (("development","market_dev_v1","calendar_development_v2","attribution_extract_v1",1),
                                        ("later","market_reserved_v1","calendar_full_v1","attribution_extract_full_v1",2)):
        files=sorted((out/label/"cells").glob("*.json"))
        if [p.stem for p in files]!=sorted(names) or len(names)!=28:raise ValueError("Incomplete required historical coverage")
        for cell in files:
            m=cell.stem;newcell=json.loads(cell.read_text());fitjson=json.loads(Path(newcell["fits"]["path_file"]).read_text())
            original=json.loads((SIBLING/f"outputs/{arch}/cells/{m}.json").read_text())["fits"]
            inputs=load_tail_inputs(ROOT/f"research/data/oxfordman_barde/{data}/{m}.npz",ROOT/f"research/outputs/{extract}/{m}.npz",scored_split=split)
            r=inputs.returns_percent;cohorts.append(dict(cohort=label,market=m,scored_days=int(inputs.scored.sum()),shared_prefix_fit=newcell["receipts"]["shared_prefix_fit"]))
            for p in newcell["receipts"]["archive_inputs"]:
                if sha(p["path"])!=p["sha256"]:raise ValueError("Original archive modified")
                preservation.append(p)
            unchanged.extend(dict(cohort=label,market=m,**x) for x in newcell["receipts"]["unchanged_daily_checks"])
            for mod in ("garch","gas"):
                groups=[(None,fitjson[mod],original[mod],"nll")] if mod=="garch" else [(a,fitjson[mod][a],original[mod][a],"fz0") for a in original[mod]]
                for alpha,newfits,oldfits,field in groups:
                    byorigin={f["t"]:f for f in newfits}
                    for old in oldfits:
                        t=old["t"];new=byorigin[t]
                        if mod=="garch":
                            params=[old[k] for k in ("omega","alpha","beta","nu")]
                            h=legacy.garch_variance(r[:t],*params[:3],float(np.mean(r[:t]**2)))
                            actual=-float(np.mean(legacy.t_loglik(r[1:t],h[1:],params[3])))
                        else:
                            params=[old[k] for k in ("a","b","beta","gamma")]
                            v,e=fast_gas_path(r[:t],*params,float(alpha))
                            actual=float(np.mean(legacy.fz0(r[50:t],v[50:],e[50:],float(alpha))))
                        if abs(actual-old[field])>1e-10:raise ValueError("Archived objective cannot be independently reproduced")
                        if abs(new["historical_replay_objective"]-actual)>1e-7:raise ValueError("Historical endpoint differs from actual archive: explicit archive seed required")
                        if not new["accepted"] or not new["selected_converged"] or new["objective"]>actual+1e-7:raise ValueError("Repaired endpoint fails actual archival quality gate")
                        rec=dict(cohort=label,market=m,model=mod,alpha=alpha,origin=t,
                                 actual_archived_objective=actual,archived_objective_reproduction_error=actual-old[field],
                                 causal_replay_objective=new["historical_replay_objective"],repaired_objective=new["objective"],
                                 training_improvement=actual-new["objective"],selected_converged=new["selected_converged"],
                                 causal_replay_selected_converged=new["historical_replay"]["selected_converged"],
                                 replay_parameter_max_abs_difference=float(np.max(np.abs(np.asarray(new["historical_replay"]["parameters"])-np.asarray(params)))))
                        records.append(rec)
            if any(row.get("substituted_scored_days",0) for row in newcell["rows"]):raise ValueError("Scored substitution remains")
    protocol=json.loads((out/"protocol.json").read_text())
    for p,h in protocol["source_sha256"].items():
        if sha(p)!=h:raise ValueError(f"Source changed during run: {p}")
    result=dict(status="passed_complete_coverage_actual_archival_quality_and_preservation_gates",cohorts=cohorts,
                refit_cohort_records=len(records),causal_historical_failures=sum(not x["causal_replay_selected_converged"] for x in records),
                repaired_nonconverged=0,scored_substitutions=0,
                max_archived_objective_reproduction_error=max(abs(x["archived_objective_reproduction_error"]) for x in records),
                max_causal_replay_objective_difference=max(abs(x["causal_replay_objective"]-x["actual_archived_objective"]) for x in records),
                max_repaired_objective_deterioration=max(-x["training_improvement"] for x in records),
                original_archive_preserved=preservation,unchanged_methods_max_daily_loss_difference=max(x["max_abs_daily_loss_difference"] for x in unchanged),
                unchanged_daily_checks=len(unchanged),records=records)
    if (out/"quality.json").exists():raise FileExistsError("Refusing quality-receipt overwrite")
    audit_fhs(out)
    save(out/"quality.json",result);print(result["status"],len(records),flush=True)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);audit(ap.parse_args().out)

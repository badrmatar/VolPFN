"""Scientifically equivalent shared-count legacy tail bootstrap and comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from .optimizer_core import ROOT, SIBLING, legacy
from .optimizer_run import save,sha

CANONICAL_MARKETS=("AEX","AORD","BFX","BSESN","BVSP","DJI","FCHI","FTSE","GDAXI","GSPTSE","HSI","IBEX",
                   "IXIC","KS11","KSE","MXX","N225","NSEI","OMXC20","OMXHPI","OMXSPI","OSEAX","RUT","SMSI",
                   "SPX","SSEC","SSMI","STOXX50E")


def validated_roster(out):
    """Standalone statistics require the canonical panel and both quality gates."""
    out=Path(out);expected=json.loads((out/"protocol.json").read_text())["markets"]
    if sorted(expected)!=list(CANONICAL_MARKETS):
        raise ValueError("Statistics require the canonical 28-market historical roster; partial/noncanonical runs cannot be complete comparisons")
    try:
        quality=json.loads((out/"quality.json").read_text())
        fhs=json.loads((out/"fhs_optimizer_quality.json").read_text())
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        raise ValueError("Passed GARCH/GAS and FHS quality receipts are required before statistics") from exc
    cohorts=[(c["cohort"],c["market"]) for c in quality.get("cohorts",[])]
    required=[(label,m) for label in ("development","later") for m in CANONICAL_MARKETS]
    if (quality.get("status")!="passed_complete_coverage_actual_archival_quality_and_preservation_gates"
            or quality.get("repaired_nonconverged")!=0 or quality.get("scored_substitutions")!=0
            or sorted(cohorts)!=sorted(required)
            or fhs.get("status")!="passed_independent_actual_FHS_optimizer_archive_gate"):
        raise ValueError("Complete passed GARCH/GAS and FHS quality gates with zero scored substitutions are required before statistics")
    return expected


def bootstrap_counts(n):
    rng=np.random.default_rng(legacy.BOOT["seed"]);reps=legacy.BOOT["reps"];block=legacy.BOOT["block"]
    counts=np.empty((reps,n),dtype=float)
    for offset in range(0,reps,100):
        starts=rng.integers(0,n,size=(100,int(np.ceil(n/block))))
        idx=((starts[:,:,None]+np.arange(block))%n).reshape(100,-1)[:,:n]
        for j,row in enumerate(idx):counts[offset+j]=np.bincount(row,minlength=n)
    return counts


def boot_result(samples,delta):
    p=min(1.,2*min(float(np.mean(samples<=0)),float(np.mean(samples>=0))))
    return dict(mean_difference=float(np.mean(delta)),interval95=np.quantile(samples,[.025,.975]).tolist(),
                p_two_sided=max(p,1/len(samples)),markets_improved=int(np.sum(delta<0)),markets=len(delta))


def all_boots(daily,dates,contrasts):
    markets=sorted(daily);union=np.array(sorted(set(np.concatenate([dates[m] for m in markets]))))
    counts=bootstrap_counts(len(union));samples=np.zeros((len(counts),len(contrasts)));delta=[]
    for m in markets:
        position=np.searchsorted(union,dates[m]);c=counts[:,position];den=c.sum(axis=1)
        if np.any(den<=0):raise ValueError("Bootstrap lost a complete market")
        values=np.column_stack([daily[m][ka]-daily[m][kb] for ka,kb in contrasts])
        samples+=(c@values)/den[:,None]
        delta.append(values.mean(axis=0))
    samples/=len(markets);delta=np.asarray(delta)
    return [boot_result(samples[:,j],delta[:,j]) for j in range(len(contrasts))]


def holm(p):
    p=np.asarray(p);order=np.argsort(p,kind="stable");out=np.empty(len(p));last=0.
    for rank,i in enumerate(order):last=max(last,min(1.,p[i]*(len(p)-rank)));out[i]=last
    return out.tolist()


def analysis(out,label):
    expected=validated_roster(out)
    root=out/label;rows,daily,dates=legacy.load_cells(root)
    required_rows={(m,fc,a) for m in expected for fc in legacy.FORECASTERS for a in legacy.ALPHAS}
    row_keys=[(r["market"],r["forecaster"],r["alpha"]) for r in rows]
    if (sorted(daily)!=sorted(expected) or sorted(dates)!=sorted(expected)
            or len(row_keys)!=len(required_rows) or set(row_keys)!=required_rows):
        raise ValueError("Incomplete or duplicated canonical 28-market/24-cell coverage")
    if any(r.get("substituted_scored_days",0) for r in rows):raise ValueError("A fully converged comparison cannot contain substituted scored dates")
    roster=[f"{fc}|{a}" for a in legacy.ALPHAS for fc in legacy.FORECASTERS]
    contrasts=[(f"{key}|{cb}",f"{key}|original") for key in roster for cb in legacy.CALIBEATERS]
    fhsk=[(f"{fc}|{a}|{cb}",f"fhs_garch|{a}|original") for a in legacy.ALPHAS for fc,cb in
          (("garch_gauss","ratio_ftl"),("garch_gauss","guard_ftl" if label=="development" else "guard_win1000"),("volpfn_t8","original"),("volpfn_t8","guard_ftl" if label=="development" else "guard_win1000"))]
    boots=all_boots(daily,dates,contrasts+fhsk);bi={c:b for c,b in zip(contrasts+fhsk,boots)}
    variant="guard_ftl" if label=="development" else "guard_win1000"
    panel={};decisions=[]
    oldrun="market_dev_v1" if label=="development" else "market_reserved_v1"
    oldpanel=json.loads((SIBLING/f"outputs/{oldrun}/panel.json").read_text())
    for key in roster:
        fc,astr=key.split("|");a=float(astr);sub=[r for r in rows if r["forecaster"]==fc and r["alpha"]==a]
        c={name:float(np.mean([r[name] for r in sub])) for name in ("original","hindsight_full","hindsight_window")}
        c.update(hit_original=float(np.mean([r["original_hit"] for r in sub])),markets=len(sub))
        for cb in legacy.CALIBEATERS:
            c[cb]=float(np.mean([r[cb] for r in sub]));c[cb+"_hit"]=float(np.mean([r[cb+"_hit"] for r in sub]))
            c[cb+"_boot"]=bi[(f"{key}|{cb}",f"{key}|original")]
        c["mcb_full"]=c["original"]-c["hindsight_full"];c["mcb_window"]=c["original"]-c["hindsight_window"]
        c["gain"]=c["original"]-c[variant]
        c["capture_full"]=c["gain"]/c["mcb_full"] if abs(c["mcb_full"])>1e-9 else None
        c["capture_window"]=c["gain"]/c["mcb_window"] if abs(c["mcb_window"])>1e-9 else None
        panel[key]=c
        for wrapper,binding in ((variant,True),("ratio_"+variant.split("_",1)[1],False),("switch_"+variant.split("_",1)[1],False),("adapt",False)):
            b=c[wrapper+"_boot"]
            if fc in legacy.MISSPECIFIED:
                ok=b["mean_difference"]<0 and b["p_two_sided"]<.05 and b["markets_improved"]>=20
                decisions.append(dict(rule="M1_gain",wrapper=wrapper,alpha=a,forecaster=fc,passed=bool(ok),informational=not binding or fc=="hist250",
                                      gain=-b["mean_difference"],p=b["p_two_sided"],markets_improved=b["markets_improved"],markets=b["markets"]))
            if fc in legacy.SPECIALISTS:
                excess=c[wrapper]-c["original"];ok=excess<=legacy.REGRET_BUDGET[a]
                decisions.append(dict(rule="M2_no_harm",wrapper=wrapper,alpha=a,forecaster=fc,passed=bool(ok),informational=not binding,
                                      excess=excess,budget=legacy.REGRET_BUDGET[a],p=b["p_two_sided"],markets_improved=b["markets_improved"],markets=b["markets"]))
            if fc in ("ewma_gauss","garch_gauss"):
                cap=(c["original"]-c[wrapper])/c["mcb_full"] if abs(c["mcb_full"])>1e-9 else None
                decisions.append(dict(rule="M3_capture",wrapper=wrapper,alpha=a,forecaster=fc,passed=bool(cap is not None and cap>=.5),informational=not binding or a<.02,capture_full=cap))
    # The original selection roster contained seven methods; later FHS is a
    # comparator addendum and does not retroactively enter development selection.
    totals={cb:float(sum(panel[k]["original"]-panel[k][cb] for k in panel if not k.startswith("fhs_garch|"))) for cb in legacy.RATIOS}
    choice=max(totals,key=totals.get)
    decisions.append(dict(rule="M4_variant_choice",chosen=choice,chosen_guard="guard_"+choice[6:],total_gain_by_variant=totals,
                          cohort=label,status=("repaired development sensitivity; historical later guard remains frozen" if label=="development"
                                               else "retrospective later diagnostic only; never a development choice or forecast selection")))
    development_choice=choice if label=="development" else next(d["chosen"] for d in json.loads((out/"development/decisions.json").read_text()) if d["rule"]=="M4_variant_choice")
    m1=[d for d in decisions if d["rule"]=="M1_gain" and not d["informational"]]
    for d,p in zip(m1,holm([d["p"] for d in m1])):d.update(p_holm=p,passed_holm=bool(d["gain"]>0 and p<.05 and d["markets_improved"]>=20))
    olddec=json.loads((SIBLING/f"outputs/{oldrun}/decisions.json").read_text())
    changed=[]
    for d in decisions:
        old=next((o for o in olddec if all(o.get(k)==d.get(k) for k in ("rule","wrapper","alpha","forecaster"))),None)
        if old is not None and "passed" in d and d["passed"]!=old["passed"]:changed.append(dict(before=old,after=d))
    scoredeltas={k:{nm:panel[k][nm]-oldpanel[k][nm] for nm in ("original",variant,"ratio_ftl","bins5") if nm in oldpanel[k]} for k in panel if k in oldpanel}
    fhs={f"{ka} minus {kb}":bi[(ka,kb)] for ka,kb in fhsk}
    save(root/"panel.json",panel);save(root/"decisions.json",decisions)
    save(root/"comparison_to_archive.json",dict(original_run=oldrun,variant=variant,score_deltas=scoredeltas,changed_decisions=changed,
                                               fhs_comparisons=fhs,binding_passed=sum(d["passed"] for d in decisions if "passed"in d and not d["informational"]),
                                               binding_total=sum("passed"in d and not d.get("informational",True) for d in decisions),
                                               historical_selection_frozen="guard_win1000",repaired_development_choice=development_choice,
                                               retrospective_later_m4_maximizer=choice if label=="later" else None))
    adaptive_run="market_dev_v2" if label=="development" else "market_reserved_v2_a05"
    adaptive_panel=json.loads((SIBLING/f"outputs/{adaptive_run}/panel.json").read_text())
    adaptive_decisions=json.loads((SIBLING/f"outputs/{adaptive_run}/decisions.json").read_text())
    adaptive_changes=[]
    for d in decisions:
        if d.get("wrapper")!="adapt":continue
        old=next((o for o in adaptive_decisions if all(o.get(k)==d.get(k) for k in ("rule","wrapper","alpha","forecaster"))),None)
        if old is not None and "passed"in d and old["passed"]!=d["passed"]:adaptive_changes.append(dict(before=old,after=d))
    save(root/"adaptive_comparison_to_archive.json",dict(original_run=adaptive_run,scope="Exploratory procedure designed after later-block exposure",
                                                       score_deltas={k:{nm:panel[k][nm]-adaptive_panel[k][nm] for nm in ("original","adapt","ratio_ftl")}
                                                                     for k in panel if k in adaptive_panel},changed_decisions=adaptive_changes))
    return dict(label=label,markets=len(daily),rows=len(rows),changed_decisions=len(changed),score_deltas=scoredeltas)


def correct_selection_labels(out):
    import shutil
    snap=out/"report_metadata_before_selection_label_fix";snap.mkdir()
    for label in ("development","later"):
        folder=snap/label;folder.mkdir()
        for name in ("decisions.json","comparison_to_archive.json"):
            shutil.copy2(out/label/name,folder/name)
    shutil.copy2(out/"summary.json",snap/"summary.json")
    development=json.loads((out/"development/decisions.json").read_text())
    devchoice=next(d["chosen"] for d in development if d["rule"]=="M4_variant_choice")
    for label in ("development","later"):
        decisions=json.loads((out/label/"decisions.json").read_text())
        m4=next(d for d in decisions if d["rule"]=="M4_variant_choice");m4["cohort"]=label
        m4["status"]=("repaired development sensitivity; historical later guard remains frozen" if label=="development" else
                      "retrospective later diagnostic only; never a development choice or forecast selection")
        save(out/label/"decisions.json",decisions)
        comp=json.loads((out/label/"comparison_to_archive.json").read_text())
        comp["repaired_development_choice"]=devchoice
        comp["retrospective_later_m4_maximizer"]=m4["chosen"] if label=="later" else None
        save(out/label/"comparison_to_archive.json",comp)
    summary=json.loads((out/"summary.json").read_text())
    summary["code_sha256"]=sha(__file__)
    summary["selection_label_correction"]="Actual development choice ratio_win1000 carried into later report; ratio_win2000 is retrospective later diagnostic only. Numeric forecasts, scores, bootstraps and decision passes preserved."
    save(out/"summary.json",summary)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--correct-selection-labels",action="store_true")
    args=ap.parse_args();out=Path(args.out)
    if args.correct_selection_labels:return correct_selection_labels(out)
    if (out/"summary.json").exists():raise FileExistsError("Refusing summary overwrite")
    result=[analysis(out,label) for label in ("development","later")]
    save(out/"summary.json",dict(status="complete_comparisons",results=result,code_sha256=sha(__file__),bootstrap_equivalence="Same legacy draws via integer date multiplicities, equal-market per-draw means"))
    print("complete comparisons",flush=True)


if __name__=="__main__":main()

"""Rebuild actual archived pairs to quantify forecast and score effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from .optimizer_core import ROOT,SIBLING,legacy,fast_gas_path
from .optimizer_run import save
from experiments.audit_support.strict_tail_inputs import load_tail_inputs


def original_paths(r,logs):
    T=len(r);h=np.empty(T);nu=np.empty(T);gas={}
    g=logs["garch"]
    for j,f in enumerate(g):
        lo=0 if j==0 else f["t"];hi=g[j+1]["t"] if j+1<len(g) else T
        x=legacy.garch_variance(r[:hi],f["omega"],f["alpha"],f["beta"],float(np.mean(r[:f["t"]]**2)))
        h[lo:hi]=np.maximum(x[lo:hi],1e-6);nu[lo:hi]=f["nu"]
    for a,lg in logs["gas"].items():
        v=np.empty(T);e=np.empty(T)
        for j,f in enumerate(lg):
            lo=0 if j==0 else f["t"];hi=lg[j+1]["t"] if j+1<len(lg) else T
            vv,ee=fast_gas_path(r[:hi],f["a"],f["b"],f["beta"],f["gamma"],float(a))
            v[lo:hi],e[lo:hi]=vv[lo:hi],ee[lo:hi]
        gas[float(a)]=(v,e)
    return dict(h_garch=h,nu_garch=nu,garch_log=g),gas


def effects(out):
    out=Path(out);records=[]
    for label,arch,data,extract,split in (("development","market_dev_v1","calendar_development_v2","attribution_extract_v1",1),
                                        ("later","market_reserved_v1","calendar_full_v1","attribution_extract_full_v1",2)):
        for cell in sorted((out/label/"cells").glob("*.json")):
            m=cell.stem;meta=json.loads(cell.read_text());fitfile=Path(meta["fits"]["path_file"])
            fits=json.loads(fitfile.read_text())
            inputs=load_tail_inputs(ROOT/f"experiments/data/oxfordman_barde/{data}/{m}.npz",ROOT/f"experiments/outputs/{extract}/{m}.npz",scored_split=split)
            r=inputs.returns_percent;T=len(r);sc=inputs.scored
            oldfits=json.loads((SIBLING/f"outputs/{arch}/cells/{m}.json").read_text())["fits"]
            oldcache,oldgas=original_paths(r,oldfits)
            fhsarch="market_dev_fhs_v1" if label=="development" else "market_reserved_fhs_v1"
            fhsfits=json.loads((SIBLING/f"outputs/{fhsarch}/cells/{m}.json").read_text())["fits"]
            fhscache,_=original_paths(r,fhsfits)
            with np.load(fitfile.with_suffix(".npz"),allow_pickle=False) as a:
                newcache=dict(h_garch=a["h_garch"][:T],nu_garch=a["nu_garch"][:T],garch_log=[f for f in fits["garch"] if f["t"]<T])
                newgas={al:(a[f"gas_v|{al}"][:T],a[f"gas_e|{al}"][:T]) for al in legacy.ALPHAS}
            saved=np.load(SIBLING/f"outputs/{arch}/cells/{m}.npz",allow_pickle=False)
            fhssaved=np.load(SIBLING/f"outputs/{fhsarch}/cells/{m}.npz",allow_pickle=False)
            for alpha in legacy.ALPHAS:
                for fc in ("garch_gauss","garch_t","gas_fz0","fhs_garch"):
                    pairs=[]
                    for cache,gas in ((fhscache if fc=="fhs_garch" else oldcache,oldgas),(newcache,newgas)):
                        vv,ee=gas[alpha] if fc=="gas_fz0" else legacy.forecast(fc,r,alpha,cache)
                        vv=np.minimum(vv,legacy.V_CAP);ee=np.minimum(ee,vv-1e-6);pairs.append((vv,ee))
                    score=[legacy.fz0(r,*p,alpha)[sc] for p in pairs]
                    key=f"{fc}|{alpha}|original"
                    actual_saved=fhssaved if fc=="fhs_garch" else saved
                    if key not in actual_saved.files or not np.allclose(score[0],actual_saved[key],rtol=1e-12,atol=1e-12):raise ValueError("Archived score path cannot be reconstructed")
                    dv=pairs[1][0][sc]-pairs[0][0][sc];de=pairs[1][1][sc]-pairs[0][1][sc]
                    records.append(dict(cohort=label,market=m,model=fc,alpha=alpha,scored_days=int(sc.sum()),
                                        max_abs_VaR_change_percent_return_units=float(np.max(np.abs(dv))),
                                        max_abs_ES_change_percent_return_units=float(np.max(np.abs(de))),
                                        max_abs_relative_VaR_change=float(np.max(np.abs(dv/pairs[0][0][sc]))),
                                        max_abs_relative_ES_change=float(np.max(np.abs(de/pairs[0][1][sc]))),
                                        mean_original_FZ0_change=float(np.mean(score[1]-score[0])),
                                        max_abs_daily_original_FZ0_change=float(np.max(np.abs(score[1]-score[0])))))
            saved.close()
            fhssaved.close()
    result=dict(status="passed_actual_archived_forecast_and_score_reconstruction",records=records)
    if (out/"forecast_effects.json").exists():raise FileExistsError("Refusing effects overwrite")
    save(out/"forecast_effects.json",result);print("forecast effects",len(records),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);effects(ap.parse_args().out)

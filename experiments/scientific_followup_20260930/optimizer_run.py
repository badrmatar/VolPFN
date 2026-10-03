"""Full historical tail regeneration with strict inputs and additive receipts."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
from .optimizer_core import ROOT, SIBLING, POLICY, legacy, fast_gas_path
from .optimizer_paths import specialist_paths
from experiments.audit_support.strict_tail_inputs import load_tail_inputs
from calibeat.adaptive import ftl_window_select


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj): Path(p).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))


def wrapped_pairs(r,v,e,alpha):
    result={"original":(v,e)}
    for cb in legacy.CALIBEATERS:
        if cb == "adapt":
            names=["ratio_ftl","ratio_win1000","ratio_win2000"]
            L=np.stack([legacy.fz0(r,v,e,alpha)]+[legacy.fz0(r,*result[n],alpha) for n in names],axis=1)
            ch=ftl_window_select(L,start=legacy.WARM+legacy.SWITCH_MIN,W=legacy.GUARD_WINDOW,margin=legacy.GUARD_MARGIN[alpha])
            vh,eh=v.copy(),e.copy()
            for j,n in enumerate(names,start=1):
                ix=ch==j; vh[ix],eh[ix]=result[n][0][ix],result[n][1][ix]
        elif cb.startswith(("switch_","guard_")):
            kind,tail=cb.split("_",1); vw,ew=result["ratio_"+tail]
            vh,eh=legacy.switch(legacy.fz0(r,v,e,alpha),legacy.fz0(r,vw,ew,alpha),v,e,vw,ew,
                              window=legacy.GUARD_WINDOW if kind=="guard" else None,
                              margin=legacy.GUARD_MARGIN[alpha] if kind=="guard" else 0.)
        else: vh,eh=legacy.calibeat(cb,r,v,e,alpha)
        result[cb]=(vh,eh)
    return result


def cell(name,out):
    clock=time.time(); out=Path(out)
    full=load_tail_inputs(ROOT/f"experiments/data/oxfordman_barde/calendar_full_v1/{name}.npz",
                          ROOT/f"experiments/outputs/attribution_extract_full_v1/{name}.npz",scored_split=2)
    dev=load_tail_inputs(ROOT/f"experiments/data/oxfordman_barde/calendar_development_v2/{name}.npz",
                         ROOT/f"experiments/outputs/attribution_extract_v1/{name}.npz",scored_split=1)
    if not np.array_equal(dev.date,full.date[:len(dev.date)]):
        raise ValueError("Development is not the identical full-calendar calendar prefix")
    same_returns=np.array_equal(dev.returns_percent,full.returns_percent[:len(dev.date)])
    r=full.returns_percent
    h,nu,glog,gfail=specialist_paths(r,"garch_t")
    gas={}; gaslog={}; gasfail={}
    for alpha in legacy.ALPHAS:
        v,e,log,fail=specialist_paths(r,"gas_fz0",alpha)
        gas[alpha]=(v,e); gaslog[str(alpha)]=log; gasfail[str(alpha)]=fail
    pathdict=dict(date=full.date,h_garch=h,nu_garch=nu,garch_substituted=gfail)
    for alpha in legacy.ALPHAS:
        pathdict[f"gas_v|{alpha}"],pathdict[f"gas_e|{alpha}"]=gas[alpha]
        pathdict[f"gas_substituted|{alpha}"]=gasfail[str(alpha)]
    (out/"paths").mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"paths"/f"{name}.npz",**pathdict)
    save(out/"paths"/f"{name}.json",dict(garch=glog,gas=gaslog,seconds=time.time()-clock))
    for label,inputs in (("development",dev),("later",full)):
        T=len(inputs.date); rr=inputs.returns_percent; sc=np.flatnonzero(inputs.scored)
        if label=="development" and not same_returns:
            hh,nn,gg,gf=specialist_paths(rr,"garch_t");gs={};ggas={};gfs={}
            for alpha in legacy.ALPHAS:
                vv,ee,ll,ff=specialist_paths(rr,"gas_fz0",alpha)
                gs[alpha]=(vv,ee);ggas[str(alpha)]=ll;gfs[str(alpha)]=ff
            pp=out/"paths"/"development";pp.mkdir(parents=True,exist_ok=True)
            dd=dict(date=inputs.date,h_garch=hh,nu_garch=nn,garch_substituted=gf)
            for alpha in legacy.ALPHAS:
                dd[f"gas_v|{alpha}"],dd[f"gas_e|{alpha}"]=gs[alpha];dd[f"gas_substituted|{alpha}"]=gfs[str(alpha)]
            np.savez_compressed(pp/f"{name}.npz",**dd)
            save(pp/f"{name}.json",dict(garch=gg,gas=ggas,reason="Independent fitting required because saved return prefix differs at floating-point precision"))
            fitpath=pp/f"{name}.json"
        else:
            hh,nn,gg,gf,gs,gfs=h[:T],nu[:T],[f for f in glog if f["t"]<T],gfail[:T],gas,gasfail
            fitpath=out/"paths"/f"{name}.json"
        cache=dict(h_ewma=legacy.ewma_variance(rr),h_garch=hh,nu_garch=nn,garch_log=gg,h_net=inputs.variance)
        rows=[]; daily={}; affected={}; unchanged=[]; oldfiles=[]
        oldrun="market_dev_v1" if label=="development" else "market_reserved_v1"
        fhsrun="market_dev_fhs_v1" if label=="development" else "market_reserved_fhs_v1"
        olddaily={name:{}};olddates={}
        for oldroot in (oldrun,fhsrun):
            with np.load(SIBLING/f"outputs/{oldroot}/cells/{name}.npz",allow_pickle=False) as a:
                if name in olddates and not np.array_equal(olddates[name],a["date"]):raise ValueError("Original/FHS dates differ")
                olddates[name]=a["date"].astype("U10")
                olddaily[name].update({k:a[k] for k in a.files if k!="date"})
        for root in (oldrun,fhsrun):
            for ext in ("json","npz"):
                p=SIBLING/f"outputs/{root}/cells/{name}.{ext}";oldfiles.append(dict(path=str(p),sha256=sha(p)))
        if not np.array_equal(olddates[name],inputs.date[sc]):raise ValueError("Archived scoring calendar differs")
        for alpha in legacy.ALPHAS:
            for fc in legacy.FORECASTERS:
                pair=tuple(x[:T] for x in gs[alpha]) if fc=="gas_fz0" else legacy.forecast(fc,rr,alpha,cache)
                v=np.minimum(pair[0],legacy.V_CAP);e=np.minimum(pair[1],v-1e-6)
                row=dict(market=name,forecaster=fc,alpha=alpha,scored_days=len(sc),hits_scored=int(np.sum(rr[sc]<=v[sc])))
                for cb,(vh,eh) in wrapped_pairs(rr,v,e,alpha).items():
                    ss=legacy.fz0(rr,vh,eh,alpha)
                    if not np.isfinite(ss).all():raise FloatingPointError("Nonfinite scored loss")
                    row[cb]=float(np.mean(ss[sc]));row[cb+"_hit"]=float(np.mean(rr[sc]<=vh[sc]))
                    row[cb+"_es_resid"]=float(np.mean(legacy.identification(rr[sc],vh[sc],eh[sc],alpha)[1]))
                    key=f"{fc}|{alpha}|{cb}";daily[key]=ss[sc]
                    if fc in ("ewma_gauss","ewma_t5","hist250","volpfn_t8") and key in olddaily[name]:
                        diff=float(np.max(np.abs(ss[sc]-olddaily[name][key])))
                        if diff>2e-12:raise ValueError(f"Unchanged method differs from archive {name} {key} {diff}")
                        unchanged.append(dict(key=key,max_abs_daily_loss_difference=diff))
                vr,er=legacy.hindsight_ratio_recalibration(rr,v,alpha)
                row["hindsight_full"]=float(np.mean(legacy.fz0(rr[sc],vr[sc],er[sc],alpha)))
                vr,er=legacy.hindsight_ratio_recalibration(rr[sc],v[sc],alpha)
                row["hindsight_window"]=float(np.mean(legacy.fz0(rr[sc],vr,er,alpha)))
                if fc in ("garch_gauss","garch_t","fhs_garch","gas_fz0"):
                    flags=gfs[str(alpha)][:T] if fc=="gas_fz0" else gf[:T]
                    row["substituted_scored_days"]=int(flags[sc].sum())
                    row["fully_converged_scored_path"]=not bool(flags[sc].any())
                rows.append(row)
        folder=out/label/"cells";folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f"{name}.npz",date=inputs.date[sc],**daily)
        save(folder/f"{name}.json",dict(rows=rows,fits=dict(path_file=str(fitpath)),
                                      receipts=dict(input=inputs.audit,archive_inputs=oldfiles,unchanged_daily_checks=unchanged,
                                                    shared_prefix_fit=label=="development" and same_returns),seconds=time.time()-clock))
    return dict(market=name,seconds=time.time()-clock,refits=len(glog),accepted_garch=sum(f["accepted"] for f in glog),
                accepted_gas=sum(f["accepted"] for ff in gaslog.values() for f in ff),
                carried_garch=int(gfail.sum()),carried_gas=sum(int(x.sum()) for x in gasfail.values()))


def dispatch(job): return cell(*job)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--markets",nargs="*");ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError("Versioned run destination must be new")
    out.mkdir(parents=True)
    names=args.markets or legacy.market_names(ROOT/"experiments/data/oxfordman_barde/calendar_development_v2")
    sources=[Path(__file__),Path(__file__).with_name("optimizer_core.py"),Path(__file__).with_name("optimizer_paths.py"),
             Path(__file__).with_name("OPTIMIZER_PROTOCOL.md"),ROOT/"experiments/audit_support/strict_tail_inputs.py",
             SIBLING/"calibeat/market.py",SIBLING/"calibeat/fz.py",SIBLING/"calibeat/calibeater.py",SIBLING/"calibeat/adaptive.py"]
    save(out/"protocol.json",dict(policy=POLICY,markets=names,alphas=legacy.ALPHAS,forecasters=legacy.FORECASTERS,
                                 calibeaters=legacy.CALIBEATERS,boot=legacy.BOOT,scope="historical expanding tail only; no Ch4 density or daily leaderboard replacement",
                                 source_sha256={str(p):sha(p) for p in sources}))
    clock=time.time();results=[]
    try:
        with mp.Pool(args.workers) as pool:
            for result in pool.imap_unordered(dispatch,[(n,str(out)) for n in names]):
                results.append(result);print(result,flush=True)
                save(out/"status.json",dict(status="running",markets=results,seconds=time.time()-clock))
        save(out/"status.json",dict(status="complete_cells",markets=results,seconds=time.time()-clock))
    except BaseException:
        (out/"failure.txt").write_text(traceback.format_exc())
        save(out/"status.json",dict(status="failed",completed=results,seconds=time.time()-clock));raise


if __name__=="__main__":main()

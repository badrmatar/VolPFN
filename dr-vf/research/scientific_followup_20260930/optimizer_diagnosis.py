"""Quantify objective/forecast effects for the audit's 24 selected fits."""
from pathlib import Path
import json
import time
import numpy as np
from .optimizer_core import ROOT, legacy, fit, fast_gas_path


def main():
    out = ROOT / "research/outputs/scientific_followup_20260930/optimizer"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "diagnosis_24fits.json"
    if target.exists():
        raise FileExistsError(target)
    rows = []; clock = time.time()
    for market in ("SPX", "FTSE"):
        m = legacy.load_market(market, ROOT / "research/data/oxfordman_barde/calendar_full_v1")
        r = m["r"]
        for cutoff in (750, int(np.searchsorted(m["date"], "2011-01-01")), int(np.searchsorted(m["date"], "2015-01-01"))):
            for model, alpha in (("garch_t", None), ("gas_fz0", .05), ("gas_fz0", .025), ("gas_fz0", .01)):
                th0, p0, old = fit(r[:cutoff], model, alpha, repaired=False)
                th1, p1, new = fit(r[:cutoff], model, alpha)
                hi = min(cutoff+250,len(r))
                if model == "garch_t":
                    h0 = float(np.mean(r[:cutoff]**2))
                    pairs = []
                    for p in (p0,p1):
                        h = legacy.garch_variance(r[:hi], *p[:3], h0)[cutoff:hi]
                        q,e = legacy.student_t_var_es(.01,p[3]); pairs.append((q*np.sqrt(h),e*np.sqrt(h)))
                    level = .01
                else:
                    pairs = [tuple(x[cutoff:hi] for x in fast_gas_path(r[:hi], *p, alpha)) for p in (p0,p1)]
                    level = alpha
                scores = [legacy.fz0(r[cutoff:hi], *p, level) for p in pairs]
                row = dict(market=market, cutoff=cutoff, last_training_date=m["date"][cutoff-1], model=model, alpha=alpha,
                           historical=old, repaired=new,
                           forecast_next_relative_change=(np.asarray(new["next_forecast"])/np.asarray(old["next_forecast"])-1).tolist(),
                           following_block_mean_fz_change=float(np.mean(scores[1]-scores[0])),
                           following_block_max_abs_var_change=float(np.max(np.abs(pairs[1][0]-pairs[0][0]))),
                           scored_for_diagnosis_only=True)
                rows.append(row)
                print(market,cutoff,model,alpha,"old_success",old["selected_converged"],"objective_gain",new["objective_improvement"],"next_change",row["forecast_next_relative_change"], flush=True)
    target.write_text(json.dumps(dict(scope="Post-exposure numerical sensitivity, not independent confirmation", elapsed_seconds=time.time()-clock, fits=rows),indent=2))


if __name__ == "__main__": main()

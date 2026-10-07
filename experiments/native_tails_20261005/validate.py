"""Read-only replay checks against the saved Chapter 7 neural-t8 comparator."""
from pathlib import Path
import numpy as np
from run import protocol,save,OUT,sha,HERE
from calibeat import market as M
from calibeat.fz import fz0,student_t_var_es


def main():
    p=protocol();checks=[];source_hashes={}
    root=Path(p['data_dir']).parents[2]/'outputs'
    for phase,folder in [('later','attribution_extract_full_v1'),('development','attribution_extract_v1')]:
        _,daily,dates=M.load_cells(p['comparator_dirs'][phase][0])
        for m in p['markets']:
            file=root/folder/(m+'.npz');source_hashes[str(file)]=sha(file)
            with np.load(file) as z:
                ix=np.searchsorted(z['date'].astype('U10'),dates[m])
                assert np.array_equal(z['date'].astype('U10')[ix],dates[m])
                r=z['r'][ix];h=z['h'][ix]
                with np.load(Path(p['data_dir'])/(m+'.npz')) as d:
                    j=np.searchsorted(d['date'].astype('U10'),dates[m]);current=d['returns'][j]*100
                    unit_error=float(np.max(np.abs(r-current)))
                    np.testing.assert_allclose(r,current,rtol=0,atol=1e-13)
                for a in p['tails']['alphas']:
                    v,e=M.scale_pair(np.sqrt(h),*student_t_var_es(a,8.))
                    v=np.minimum(v,M.V_CAP);e=np.minimum(e,v-1e-6)
                    old=fz0(r,v,e,a)
                    error=float(np.max(np.abs(old-daily[m][f'volpfn_t8|{a}|original'])))
                    return_effect=float(np.max(np.abs(old-fz0(current,v,e,a))))
                    assert error<1e-10 and return_effect<1e-10,(phase,m,a,error)
                    checks.append(dict(panel=phase,market=m,alpha=a,max_score_error=error,
                                       max_return_error=unit_error,max_score_effect_of_return_roundoff=return_effect))
    save(OUT/'archived_t8_replay_check.json',dict(checks=checks,all_passed=True,validation_inputs=source_hashes,
                                               validation_source_sha256=sha(HERE/'validate.py')))
    print('Archived t8 replay and units checked:',len(checks),'cells')


if __name__=='__main__':
    main()

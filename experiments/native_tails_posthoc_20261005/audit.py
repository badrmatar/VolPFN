"""Inspect invalid forecast paths and verify the completed result without refitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from models import ORIGINAL_PATH
from run import protocol,OUT,HERE,save,sha


def main():
    p=protocol();failures=[];caps=[]
    for name in p['markets']:
        log=json.loads((OUT/'fits'/(name+'.json')).read_text())
        caps.extend(dict(market=name,**row) for row in log['caps'])
        for a,entry in log['es_caviar'].items():
            state=log['states'][f'es_caviar|{a}']
            if state['status']=='complete':continue
            record=dict(market=name,model='ES-CAViaR',alpha=float(a),reason=state['reason'],optimizer_status=entry['status'])
            if entry.get('parameters') is not None:
                with np.load(Path(p['data_dir'])/(name+'.npz')) as z:
                    r=z['returns'].astype(float)*100;dates=z['date'].astype('U10');mask=z['split']==2
                th=entry['parameters'];q=ORIGINAL_PATH(r,*th[:3],entry['q0']);e=(1+np.exp(th[3]))*q
                invalid=mask&(~np.isfinite(q)|~np.isfinite(e)|(q>=0)|(e>=q))
                idx=np.flatnonzero(invalid)
                record.update(invalid_scored_days=len(idx),scored_days=int(mask.sum()),first_invalid_date=str(dates[idx[0]]),
                              first_invalid_var=float(q[idx[0]]),first_invalid_es=float(e[idx[0]]),
                              selected_start=entry['selected_start'],selected_optimizer_converged=entry['candidates'][entry['selected_start']]['success'])
            failures.append(record)
    pd.DataFrame(failures).to_csv(OUT/'invalid_forecast_paths.csv',index=False)
    pd.DataFrame(caps).to_csv(OUT/'cap_audit.csv',index=False)
    rows=json.loads((OUT/'results.json').read_text())
    adjusted=multipletests([r['holm_family_input'] for r in rows],method='holm')[1]
    for r,expected in zip(rows,adjusted):
        if r['status']=='complete':np.testing.assert_allclose(r['holm_p'],expected,rtol=0,atol=1e-15)
    seal=json.loads((HERE/'source_seal.json').read_text())
    assert all(sha(HERE/name)==digest for name,digest in seal['files'].items())
    save(OUT/'audit.json',dict(source_seal_unchanged=True,post_hoc_protocol_unchanged=True,independent_holm_check=True,
                              unavailable_comparisons=sum(r['status']!='complete' for r in rows),invalid_forecast_cells=failures,
                              original_files_unchanged=len(p['protected_sha256']),cap_counts=pd.DataFrame(caps)[['var_caps','es_caps']].sum().to_dict()))
    print(json.dumps(failures,indent=2));print('CAP TOTALS',pd.DataFrame(caps)[['var_caps','es_caps']].sum().to_dict())


if __name__=='__main__':main()

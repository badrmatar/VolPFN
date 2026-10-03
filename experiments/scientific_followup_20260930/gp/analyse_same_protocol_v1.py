"""Future strict analysis of a complete, separately versioned likelihood-v1 run.

This script cannot fill missing cells or reuse diagnostic indexing results.
The full same-protocol run has not been completed in this follow-up.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.gpvol_full import analyse_addendum_logmoments as original_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family34',action='store_true',help='separate sensitivity with original 16 competitors')
    a=p.parse_args()
    manifest=json.loads((a.run_dir/'manifest.json').read_text())
    if not manifest.get('complete_primary_roster') or manifest.get('particles')!=200:
        raise ValueError('A complete prescribed 28-market, two-seed, 200-particle run is required')
    if manifest.get('failed_cells',1)!=0:
        raise ValueError('Incomplete or failed execution cannot produce a ranking')
    stats=original_analysis.load_stats()
    bundle=json.loads((original_analysis.FRONTIER/'bundle.json').read_text())
    records=original_analysis.load_cells(original_analysis.FRONTIER,'volpfn')
    model='gpvol_likelihood_v1'
    records+=original_analysis.load_cells(a.run_dir,model)
    specs={'volpfn':bundle['model_specs']['volpfn'],model:original_analysis.gp_spec([1,2],model)}
    if a.family34:
        for name,spec in bundle['model_specs'].items():
            if name!='volpfn':
                records+=original_analysis.load_cells(original_analysis.FRONTIER,name)
                specs[name]=spec
    archive={}
    result=stats.analyse(records,specs,bundle=dict(bundle,settings=dict(
        resamples=20000,blocks=[60,20,120],bootstrap_seed=2026092601)),archive=archive)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'results.json').write_text(json.dumps(result,default=float,allow_nan=False)+'\n')
    np.savez_compressed(a.out/'bootstrap_archive.npz',**archive)
    (a.out/'provenance.json').write_text(json.dumps(dict(
        model=model,run_manifest=manifest,run_manifest_sha256=original_analysis.sha(a.run_dir/'manifest.json'),
        runner_sha256=original_analysis.sha(Path(__file__)),
        family='34-slot sensitivity' if a.family34 else '2-slot addendum primary',
        independent_future_confirmation=False),indent=2)+'\n')


if __name__=='__main__':
    main()

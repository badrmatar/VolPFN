"""Prepare a target-defined longer panel without computing forecast losses."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from research.combined_leaderboard.common import ROOT, sha, read, save, now

OUT = ROOT / 'research/outputs/frontier_extension_20260926_v1'
START, END = '2019-01-01', '2022-02-25'
SEEDS = [8101,8102,8103,8104,8105]
CLASSICAL = ['hist250','ewma','garch_t','gjr_garch_t','egarch_t','aparch_t','compact9','univol','har','loghar']
FOUNDATION = ['chronos2','moirai2','tabpfn_ts']
NEW = ['ttm_rk','ms_har','log_rhar']

def runtime_requirements():
    """Read existing environments; never install or discover different weights."""
    configurations = {
        'classical': (Path('/Users/badrmatar/Downloads/Study/Dissertation Masters/Code/Diss_Project/venv/bin/python'),
                      ['numpy', 'scipy', 'pandas', 'arch', 'statsmodels']),
        'frontier': (ROOT/'research/environments/frontier_v1/bin/python',
                     ['numpy', 'scipy', 'pandas', 'torch', 'transformers', 'chronos-forecasting']),
        'moirai': (ROOT/'research/environments/moirai_v1/bin/python', ['numpy', 'scipy', 'pandas', 'torch', 'uni2ts']),
        'ttm': (ROOT/'research/outputs/return_only_benchmark_setup_v1/foundation_venv/bin/python',
                ['numpy', 'scipy', 'pandas', 'torch', 'transformers', 'granite-tsfm']),
    }
    environments = {}
    for name, (executable, packages) in configurations.items():
        code = ('import json,platform,importlib.metadata as m; '
                'print(json.dumps(dict(python=platform.python_version(),packages={k:m.version(k) for k in '
                + repr(packages) + '})))')
        value = json.loads(subprocess.run([str(executable), '-B', '-c', code],
                          check=True, capture_output=True, text=True).stdout)
        environments[name] = dict(value, executable=str(executable), device='cpu', threads=2)
    old = read(ROOT/'research/outputs/combined_leaderboard_v2/FREEZE.json')
    for name, version in old['neural_runtime']['packages'].items():
        if environments['frontier']['packages'][name] != version:
            raise ValueError('VolPFN runtime changed from the passing bridge: '+name)
    if environments['frontier']['python'] != old['neural_runtime']['python']:
        raise ValueError('VolPFN Python patch version changed from passing bridge')
    for name, version in old['classical_runtime'].items():
        if environments['classical']['packages'][name] != version:
            raise ValueError('Old classical runtime changed: '+name)
    families = {**{m:'classical' for m in CLASSICAL + ['ms_har', 'log_rhar', 'loghar_meanlog']},
                'volpfn':'frontier', 'chronos2':'frontier', 'moirai2':'moirai',
                'ttm_rk':'ttm', 'ttm_sqrt_rk':'ttm'}
    return {model:dict(environments[environment], **({} if model.startswith('ttm_') else {'batch_size':1}))
            for model, environment in families.items()}

def model_specs():
    """Explicit moment semantics and retained seeds; diagnostics never add tests."""
    from research.combined_leaderboard.pipeline import specs as old_specs
    specs = old_specs()
    for name in FOUNDATION:
        specs[name]['output_adapter'] = specs[name]['output_adapter'].replace('Cached ', '')
    adapters = {
        'ttm_rk': 'Frozen TTM 512-96-ft-r2.1: direct RK mean point, causal 512-row scaling; zero mean, variance=M2=positive point. Benchmark adaptation, not the paper sqrt-RV recipe.',
        'ms_har': 'Two-state sqrt-RK HAR; mixture E[sqrt(RK)^2], including emission variance; zero-mean return variance/M2 adapter.',
        'log_rhar': 'Mean-log-RK HAR with Gaussian MA(1) innovations; exp(log forecast plus one-half forecast-error variance), zero mean.',
        'loghar_meanlog': 'Diagnostic theta=0 OLS mean-log-RK HAR with Gaussian MLE residual variance; zero mean.',
        'ttm_sqrt_rk': 'Diagnostic frozen TTM on sqrt RK; squared nonnegative SD point, not E[RK]; zero mean.',
    }
    for name, adapter in adapters.items():
        specs[name] = dict(family='R', output_adapter=adapter, seeds=[None],
            information_group='Lagged realised kernel (richer information)')
    for name, spec in specs.items():
        spec['primary'] = name not in ('loghar_meanlog', 'ttm_sqrt_rk')
        spec['zero_mean'] = name not in ('volpfn', 'hist250', 'garch_t', 'gjr_garch_t', 'egarch_t', 'aparch_t')
    return specs

def prepare(out=OUT):
    out=Path(out)
    if (out/'bundle.json').exists():
        raise FileExistsError('Preserve existing preparation')
    oldpath=ROOT/'research/outputs/combined_leaderboard_v2/bundle.json'
    old=read(oldpath)
    inputs={str(oldpath.relative_to(ROOT)):sha(oldpath)}
    def register(p):
        p=Path(p); inputs[str(p.relative_to(ROOT))]=sha(p)
    register(ROOT/'research/outputs/combined_leaderboard_v2/FREEZE.json')
    rows=[dict(r) for r in old['rows'] if r['phase']=='calibration']
    for name in old['markets']:
        p=ROOT/old['market_paths'][name]; register(p)
        expected=old['inputs'][str(p.relative_to(ROOT))]
        if inputs[str(p.relative_to(ROOT))]!=expected: raise ValueError('Market identity changed')
        with np.load(p,allow_pickle=False) as z:
            dates=z['date'].astype('U10'); r=z['r']; rk=z['rk']
            if not np.isfinite(r).all() or np.any(dates[1:]<=dates[:-1]): raise ValueError('Invalid history')
            for j in np.flatnonzero((dates>=START)&(dates<=END)):
                if j<1024: raise ValueError('Insufficient return context')
                good=bool(np.isfinite(rk[j]) and rk[j]>0)
                rows.append(dict(market=name,date=str(dates[j]),phase='evaluation',index=int(j),
                    r2=float(r[j])**2,rk=float(rk[j]) if good else None,primary_eligible=good))
    rows.sort(key=lambda r:(r['phase']!='calibration',r['date'],r['market']))
    for path in old['checkpoints'].values(): register(ROOT/path)
    # Checkpoint registry and compact9's fixed fitted coefficients are executable inputs.
    for suffix in ('json', 'sha256'):
        register(ROOT/f'research/outputs/return_only_benchmark_setup_v1/checkpoints/CHECKPOINT_REGISTRY.{suffix}')
    from research.evaluation_expansion.compact import dependency_paths
    for p in dependency_paths(): register(p)
    registry=ROOT/'research/outputs/frontier_development_comparison_v1/manifest.json'
    register(registry); models=read(registry)['models']; foundations={}
    for name in FOUNDATION:
        m=models[name]
        path=Path(m['path']); foundations[name]=dict(m,path=str(path.relative_to(ROOT)))
        for p,digest in m['provenance']['hashes'].items():
            p=Path(p)
            if sha(p)!=digest: raise ValueError('Foundation identity changed')
            register(p)
    ttm=ROOT/'research/outputs/return_only_benchmark_setup_v1/foundation_cache/ttm/3442406f106122f414a11f72b4ebfe4878bec89d'
    for p in ttm.iterdir():
        if p.is_file(): register(p)
    specs=model_specs()
    ev=[r for r in rows if r['phase']=='evaluation']
    bundle=dict(created_utc=now(),kind='exposed_historical_longer_panel_rerun',start=START,end=END,
        rows=rows,markets=old['markets'],market_paths=old['market_paths'],checkpoints=old['checkpoints'],
        inputs=inputs,foundation_models=foundations,ttm_path=str(ttm.relative_to(ROOT)),model_specs=specs,
        runtime_requirements=runtime_requirements(),
        counts=dict(S_all=len(ev),S=sum(r['primary_eligible'] for r in ev),calibration=72,
                    markets=len(old['markets']),union_dates=len({r['date'] for r in ev})),
        missing_rk=[dict(market=r['market'],date=r['date']) for r in ev if not r['primary_eligible']],
        exposure='Previously inspected historical period; causal test-time inputs do not make recent foundation weights historically available. This is not untouched confirmation.',
        units=old['units'],calibration=old['calibration'],source_bundle_sha256=sha(oldpath),
        source_bundle_path=str(oldpath.relative_to(ROOT)),
        bootstrap=dict(resamples=20000,main_block=60,sensitivity_blocks=[20,120],seed=2026092601),
        settings=dict(resamples=20000,blocks=[60,20,120],bootstrap_seed=2026092601,
                      primary_family_slots=32),
        budget=dict(max_usd=10,max_total_hours=8,stop_not_terminate=True,
                    absolute_stop_utc='2026-09-26T10:00:00+00:00'))
    save(out/'bundle.json',bundle)
    save(out/'PREPARATION.json',dict(status='prepared_not_frozen',counts=bundle['counts'],bundle_sha256=sha(out/'bundle.json')))
    print(json.dumps(bundle['counts']))
    return bundle

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    prepare(p.parse_args().out)

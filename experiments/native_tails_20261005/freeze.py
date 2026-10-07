"""Freeze this comparison before any native-tail scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO.parent / 'dr-vf'
SCI = Path('/Users/badrmatar/.codex/worktrees/e786/Diss_Project/dr-vf')
TAIL = REPO.parent / 'calibeat-var-es'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    path = HERE / 'protocol.json'
    if path.exists():
        raise FileExistsError('Protocol already frozen')
    registry_path = SCI / 'research/outputs/return_only_benchmark_setup_v1/checkpoints/CHECKPOINT_REGISTRY.json'
    registry = json.loads(registry_path.read_text())
    assert sha(registry_path) == registry_path.with_suffix('.sha256').read_text().strip()
    checkpoints = []
    for row in registry['entries']:
        cp = SCI / row['checkpoint']
        assert sha(cp) == row['checkpoint_sha256']
        checkpoints.append(dict(seed=row['seed'], path=str(cp), sha256=sha(cp)))
    assert sorted(r['seed'] for r in checkpoints) == list(range(8101, 8106))
    panel = json.loads((TAIL/'outputs/market_reserved_v1/protocol.json').read_text())
    markets = panel['markets']
    comparator_dirs = {phase: [str(TAIL/'outputs'/d) for d in dirs] for phase, dirs in {
        'later':['market_reserved_v1','market_reserved_fhs_v1'],
        'development':['market_dev_v1','market_dev_fhs_v1']}.items()}
    data = ORIGINAL/'research/data/oxfordman_barde/calendar_full_v1'
    protected = [registry_path] + [Path(c['path']) for c in checkpoints]
    protected += [data/(m+'.npz') for m in markets]
    for dirs in comparator_dirs.values():
        for d in dirs:
            protected += list((Path(d)/'cells').glob('*.npz')) + list((Path(d)/'cells').glob('*.json'))
            protected.append(Path(d)/'protocol.json')
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    protected += [REPO/p for p in tracked if p]
    source_identity = {str(p):sha(p) for p in sorted(set(protected))}
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), name='Corrected VolPFN native five-density tail comparison',
        prior_exposure='The later panel (2015–February 2022) was inspected before this study. This is a new frozen comparison on an exposed panel, not an untouched confirmation.',
        seeds=list(range(8101,8106)), checkpoints=checkpoints, checkpoint_family='Corrected broad final checkpoints from the GJR-repaired five-seed study, 20,000 updates; not the distinct component-ablation full models or historical-trained models.',
        inference=dict(L=1024,stride=512,min_history=1,batch_windows=4,device='mps',precision='float32; raw-unit mixture arithmetic float64',
                       convention='Same within-window normalisation and overlapping-window ownership as programme_closure/evaluate.py. Each target t uses only positions before t. Regenerate mixture parameters only, with fixed weights; no training, tuning or scale calibration.',
                       prefix='No forecast at index zero. Retain native forecasts from index one to initialise the online guard; all scored dates have at least 100 previous returns.'),
        combination='Normalise each seed component weights to unit sum, then divide by five and concatenate the four components per seed into one 20-component density. Do not average VaR, ES or scores.',
        tails=dict(alphas=[.05,.025,.01],sign='Lower-return tail; negative VaR and ES',root='Bracket by min/max individual component alpha-quantiles and bisect the combined CDF; tolerance 1e-11 in raw percentage-return units.',
                   es='Sum w*(mu*T_nu(z) - scale*(nu+z*z)/(nu-1)*t_nu(z)), divided by alpha; Gaussian branch uses mu*Phi(z)-scale*phi(z).',
                   validation='Single Gaussian and Student-t unit tests before scoring; numerical integration checks on deterministic first/middle/last eligible rows of each market at every alpha. CDF residual <1e-9, ES integration tolerance 2e-7 absolute plus 2e-7 relative.'),
        markets=markets,data_dir=str(data),comparator_dirs=comparator_dirs,
        panels=dict(later='Saved split 2 dates, 2015-01-01 through 2022-02-28',development='Saved split 1 dates, 2011-01-01 through 2014-12-31'),
        scoring=dict(implementation='calibeat.fz.fz0',units='100 times decimal open-to-close log returns (percentage units)',es_floor=1e-8,var_cap=-1e-3,es_gap=1e-6,
                     comparator_policy='Use archived Chapter 7 original daily scores, with their historical optimiser acceptance policy; do not refit or silently replace them with convergence-checked sensitivity results.',
                     matching='Exact market/date alignment across native, guard, GARCH FHS, GARCH-t, GAS and neural t8; require all 28 markets and all saved panel dates. No silent row dropping.',
                     mean='Mean within market, then equal mean over 28 markets.',bootstrap={'implementation':'calibeat.market.paired_block','seed':927601,'reps':2000,'block':60,'interval':'Marginal percentile 95%','p':'Two-sided fraction across zero, minimum 1/2000; same original implementation and joint calendar resampling.'}),
        primary={'panel':'later','contrasts':['native_minus_fhs_garch','native_minus_garch_t'],'levels':[.05,.025,.01],'multiplicity':'Holm across exactly six comparisons','beats':'Negative mean and Holm p <0.05, unless the 95% interval includes zero, which is a tie. Positive significant intervals are labelled worse; otherwise no adjusted evidence.'},
        secondary={'multiplicity':'Unadjusted, descriptive/exploratory','contrasts':['native_minus_gas_fz0','native_minus_volpfn_t8','guard_minus_native'],
                   'development':'All five contrasts at every level, unadjusted.',
                   'diagnostics':'Native and native guard: per-market hit counts/rates, central binomial 95% band using n=int(mean matched panel days), as Chapter 7; also actual-market-n band. Report Christoffersen independence LR (1 df) and joint conditional-coverage LR (2 df), unadjusted p<0.05, rejection counts out of 28; use <= for hits and successive observed market sessions.'},
        guard={'implementation':'calibeat.market.calibeat(ratio_win1000) plus market.switch','residual_window':1000,'comparison_window':1000,'warm':250,'minimum_comparisons':250,'margin':{'.05':.00175,'.025':.003,'.01':.0065},
               'initialisation':'Run causally from the first available native forecast (index one); no invented day-zero prediction or fallback. This differs by one initial observation from historical comparators. Keep and disclose it. All scored days follow warmup.'},
        deviations='Record any deviations or numerical/backend limitations separately. Never edit this frozen protocol. Stop on unavailable checkpoints or incomplete/nonfinite matched forecasts.',
        protected_sha256=source_identity)
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    (HERE/'protocol.sha256').write_text(sha(path)+'\n')
    print('Frozen',sha(path),'protected files',len(source_identity),flush=True)


if __name__ == '__main__':
    freeze()

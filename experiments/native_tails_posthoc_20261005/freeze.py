"""Freeze the separate, explicitly post-hoc nine-comparison study."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PREVIOUS=REPO/'experiments/native_tails_20261005'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=HERE/'POST_HOC_PROTOCOL.json'
    if path.exists():raise FileExistsError('Post-hoc protocol already frozen')
    old=json.loads((PREVIOUS/'protocol.json').read_text())
    copy=HERE/'legacy_gp_fz';copy.mkdir()
    gp=REPO.parent/'gp-fz'
    for name in ('benchmark_es_caviar.py','gp_fz_phase1.py','gp_fz_phase0.py'):
        shutil.copyfile(gp/name,copy/name)
        assert sha(gp/name)==sha(copy/name)
    protected={}
    for p in PREVIOUS.rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    for p in gp.glob('*.py'):protected[str(p)]=sha(p)
    for name in subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0'):
        if name:protected[str(REPO/name)]=sha(REPO/name)
    for m in old['markets']:
        file=Path(old['data_dir'])/(m+'.npz');protected[str(file)]=sha(file)
    p=dict(label='POST-HOC',created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Nine new comparisons only. The panel and preceding native-tail results have already been inspected. This is not a new untouched confirmation.',
        previous_protocol=str(PREVIOUS/'protocol.json'),previous_protocol_sha256=sha(PREVIOUS/'protocol.json'),
        native_tails_dir=str(PREVIOUS/'results/tails'),data_dir=old['data_dir'],markets=old['markets'],
        panel='Later panel only: identical saved split-2 market dates from 2015 to February 2022; exactly 28 markets and 50,073 matched observations.',
        alphas=[.05,.025,.01],models=['gjr_garch_t','fhs_gjr_garch','es_caviar'],
        native='Reuse the completed native five-density VaR/ES forecasts exactly; no inference rerun, training, recalibration or changes to earlier results.',
        gjr=dict(source='experiments/tier_a_confirmation/econometric.py:forecast_econometric; unchanged default GJR specification',
                 mean='Constant',vol='GARCH',p=1,o=1,q=1,distribution='variance-standardised Student-t',history_window=1000,refit_every=20,
                 first_origin='First scored later-panel observation of each market',optimizer='One arch default start per scheduled fit, maxiter=1000; existing strict convergence/parameter checks.',
                 failure_policy='A rejected refit carries the previous accepted parameters, with no rescue fits. An initial failure or invalid daily filter makes that market/model unavailable. Record every failed attempt.',
                 tails='mu + sqrt(variance) times the unit-variance Student-t lower-tail VaR/ES.'),
        fhs=dict(scale='Same accepted GJR-t fits, fitted constant mean, conditional variance and refit schedule as the new GJR-GARCH-t comparator.',
                 residuals='At each accepted fit, compute (r-mu)/conditional_sigma on that 1000-return training window, excluding its first observation as in Chapter 7 FHS. Use the existing empirical_var_es fractional-boundary rule.',
                 forecast='Freeze residual quantile/ES until the next accepted refit; update conditional scale causally each day. Add fitted mean to both tail values. If a GJR refit fails, retain the previous accepted residual pair and parameters.',
                 failure_policy='Any invalid accepted-fit residual pair or incomplete scored path makes this market/model unavailable; no silent repair.'),
        es_caviar=dict(source=str(gp/'benchmark_es_caviar.py'),source_sha256=sha(gp/'benchmark_es_caviar.py'),
                       variant='Original SAV quantile recursion with multiplicative ES; use fit_es_caviar and caviar_path verbatim.',n_fit=1000,seed=0,
                       estimation='Fit once on first 1000 returns of each full market history at each alpha; recurse forward without refitting or retuning.',
                       optimizer='Original 30 random starts, original ranges, Nelder-Mead, maxiter=4000, xatol=fatol=1e-6. Select strictly smallest objective exactly as original code.',
                       scoring='Ignore old helper score summaries; evaluate issued raw VaR/ES with unchanged Chapter 7 FZ0.',
                       acceleration='Optional Numba compilation of the unchanged caviar_path only, without fastmath/parallelism, after exact synthetic-path and objective parity tests. This does not alter starts, selection or optimiser settings.',
                       failure_policy='Do not replace the legacy-selected candidate. If it is nonconverged, has invalid/sentinel objective, nonfinite parameters or invalid scored tail path, mark market/level unavailable and report it. Log all 30 candidate diagnostics.'),
        comparisons=dict(direction='native VolPFN minus new comparator; negative favours native',family_size=9,
                         holm='Holm across exactly these nine tests, separate from every preceding comparison.',
                         beats='Holm p < 0.05 and negative mean difference; a 95% interval including zero takes precedence and is a tie.',
                         worse='Positive difference with Holm p <0.05 and interval entirely above zero; otherwise no adjusted evidence.',
                         missing='Never omit failed markets to obtain a smaller-panel headline. Any incomplete 28-market comparison is unavailable, with descriptive estimate/interval/raw p withheld; use p=1 solely as its reserved Holm-family input.'),
        scoring=old['scoring'],matching='Use every existing later-panel native date, exact return/date equality with the saved native tails, and no row dropping. Reuse FZ0 E_FLOOR=1e-8, VaR cap=-1e-3, ES<=VaR-1e-6, percent log-return units, equal-market means and original paired_block bootstrap: seed927601, 2000 reps, circular 60-date blocks.',
        output='One nine-row results table and complete fit-failure diagnostics, plus new code/forecasts/provenance. No changes to earlier files or dissertation text.',
        protected_sha256=protected)
    path.write_text(json.dumps(p,indent=2)+'\n');(HERE/'POST_HOC_PROTOCOL.sha256').write_text(sha(path)+'\n')
    print('POST-HOC rules frozen',sha(path),'protected files',len(protected))


if __name__=='__main__':main()

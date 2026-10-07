"""Present frozen comparisons without treating secondary p-values as Holm evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
NAMES={'fhs_garch':'GARCH FHS','garch_t':'GARCH-t','gas_fz0':'GAS','volpfn_t8':'archived neural t8','native':'native VolPFN'}


def table(frame):
    lines=['comparison | level | mean difference | 95% interval | raw p | Holm p | finding']
    for row in frame.itertuples():
        left,right=row.contrast.split('_minus_')
        label=('guard minus native' if left=='guard' else 'native minus '+NAMES[right])
        hp=f'{row.holm_p:.4f}' if pd.notna(row.holm_p) else 'not adjusted'
        finding=row.decision
        if not row.primary:
            finding={'beats':'lower FZ0 (unadjusted)','worse':'higher FZ0 (unadjusted)'}.get(finding,finding)
        lines.append(f'{label} | {100*row.alpha:g}% | {row.mean_difference:+.6f} | [{row.ci_low:+.6f}, {row.ci_high:+.6f}] | {row.raw_p:.4f} | {hp} | {finding}')
    return '\n'.join(lines)


def main():
    p=json.loads((HERE/'protocol.json').read_text())
    df=pd.read_csv(OUT/'comparisons.csv')
    # The scorer's secondary flag denotes a raw-p comparison, never a Holm claim.
    public=df.copy()
    for old,new in [('beats','lower FZ0 (unadjusted)'),('worse','higher FZ0 (unadjusted)')]:
        public.loc[(~public.primary)&(public.decision==old),'decision']=new
    public.to_csv(OUT/'results_table.csv',index=False)
    diag=pd.read_csv(OUT/'diagnostic_summary.csv')
    quad=pd.read_csv(OUT/'es_quadrature_checks.csv')
    audit=pd.read_csv(OUT/'numerical_audit.csv')
    dates=pd.read_csv(OUT/'matched_dates.csv')
    replay=json.loads((OUT/'archived_t8_replay_check.json').read_text())
    seal=json.loads((HERE/'source_seal.json').read_text())
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in seal['files'].items())
    lines=[
        'VolPFN native-tail comparison — 5 October 2026',
        '',
        'Five of six later-panel primary comparisons meet the frozen Holm rule. Native VolPFN has lower FZ0 than GARCH-t at all three levels, and than GARCH FHS at 5% and 2.5%. Against FHS at 1%, the interval includes zero: a tie, not equivalence.',
        '',
        'The later panel was inspected before this study. These results do not constitute an untouched out-of-sample confirmation.',
        '',
        'MODEL AND MATCHING',
        'Five corrected broad-prior final checkpoints (seeds 8101–8105, 20,000 updates), identified by the existing repaired-study registry. No training or calibration. These are not the separately trained component-ablation full checkpoints or the historical-trained follow-up models.',
        'Parameters regenerated using the frozen corrected overlapping-window convention: L=1024, stride=512, causal running nonzero RMS normalisation, fixed checkpoint weights. At target t, the latest feature input is return t-1. This is the matched-prior window-ownership procedure, not the distinct daily benchmark’s resetting per-origin context.',
        'The forecast is one equally weighted density average over five seeds: 20 Student components. Mixture VaR is a root of that density’s CDF; ES is its closed-form lower partial expectation divided by alpha. No averaging of quantiles, ES values, or seed scores.',
        'All 28 markets and all archived panel dates match. No rows were dropped. Returns are open-to-close log returns in percentage units. The point statistic weights markets equally.',
    ]
    for phase,g in dates.groupby('panel'):
        lines.append(f'{phase}: {int(g.days.sum()):,} market–date observations; {int(g.days.min())}–{int(g.days.max())} per market; {g.first_date.min()} through {g.last_date.max()}.')
    lines += ['', 'PRIMARY: LATER PANEL, HOLM ACROSS EXACTLY SIX',
        'All differences are native minus comparator; negative favours native. Intervals are marginal 95% calendar-block intervals, not multiplicity-adjusted intervals. Holm p-values provide the six-test adjustment. An interval containing zero is a tie.',
        table(public[public.primary]),
        '', 'SECONDARY: LATER PANEL, UNADJUSTED',
        table(public[(public.panel=='later')&~public.primary]),
        '', 'SECONDARY: DEVELOPMENT PANEL, UNADJUSTED',
        table(public[public.panel=='development']),
        '', 'COVERAGE DIAGNOSTICS, UNADJUSTED',
        'The hit-frequency screen uses the historical panel-n convention: n=int(mean matched days), 1,788 later and 1,003 development. The per-market file also reports actual-market-n bands. A hit-band pass is not a conditional-coverage result.',
        'Both Christoffersen independence (LR, 1 df) and joint conditional coverage (LR_uc+LR_ind, 2 df) are reported to make the meaning of “Christoffersen” explicit. Counts below use unadjusted p<0.05 out of 28 markets.',
        'panel | method | level | inside hit band /28 | independence rejections /28 | conditional-coverage rejections /28']
    for r in diag.itertuples():
        lines.append(f'{r.panel} | {r.method} | {100*r.alpha:g}% | {r.in_hit_band} | {r.christoffersen_ind_rejections} | {r.christoffersen_cc_rejections}')
    lines += ['', 'INTERPRETATION',
        'The native guard’s later 2.5% FZ0 difference is positive with an interval excluding zero (unadjusted p=0.017); 5% and 1% intervals include zero. Improved hit-frequency or conditional-coverage counts do not imply an improved joint VaR/ES score.',
        'The archived neural t8 comparator uses the older retained neural variance path. Native versus archived neural t8 compares complete forecasting procedures, including different checkpoint and normalisation choices. It does not isolate the effect of changing only the tail conversion.',
        '', 'PROTOCOL ADHERENCE AND IMPLEMENTATION QUALIFICATIONS',
        'No changes to the frozen protocol, primary family, thresholds, panels, comparator policy, bootstrap or guard settings. No data-driven model or rule selection was made after native scoring.',
        'The new experiment uses local MPS float32 inference, declared in the protocol; historical checkpoint training/evaluation used its recorded backend. A CPU-versus-MPS comparison passed at rtol=2e-4, atol=2e-5 for every mixture parameter over the first 1,100 AEX observations (seed 8101). This is a sampled numerical check, not a claim of bit-identical cross-device inference.',
        'Native forecasts start at index one because no history-only forecast exists at index zero. The guard starts from that first forecast, with the existing 250-observation warmup, 1,000-residual window, 1,000-score comparison window, 250 minimum comparisons and level-specific margins. The one-observation initialisation difference was fixed before scoring; no day-zero forecast was invented.',
        'Comparators reuse the Chapter 7 historical optimiser acceptance policy. The separately reported convergence-checked sensitivity was not substituted or rerun.',
        'Archived development returns differed from the current input in one checked cell by floating-point roundoff only: maximum '+format(max(x['max_return_error'] for x in replay['checks']),'.3g')+' percentage-return units; maximum effect on replayed t8 FZ0 was '+format(max(x['max_score_effect_of_return_roundoff'] for x in replay['checks']),'.3g')+'. Exact calendar matching was retained.',
        'Secondary findings are labelled lower/higher FZ0 (unadjusted); only the six primary comparisons are described as meeting the Holm “beats” rule. Intermediate comparisons.csv retains the scorer’s raw-p decision labels with primary=False; use results_table.csv for the reporting labels.',
        '', 'VERIFICATION',
        'Eight unit tests passed, including single Gaussian, single Student-t, mixed-law numerical ES, equal-density averaging, causality/original-inference parity, Holm, coverage boundaries and guard causality.',
        f'{len(quad)} real-mixture ES quadrature checks passed; maximum absolute difference {quad.absolute_error.max():.3g}. Maximum scored CDF residual {audit.max_cdf_residual.max():.3g}.',
        f'Native scored cap activations: {int(audit.var_caps.sum())} VaR and {int(audit.es_caps.sum())} ES. The original FZ0 ES floor remains 1e-8.',
        'All 168 archived neural-t8 market/level/panel score cells replayed within tolerance, confirming units and the original score implementation.',
        'Bootstrap is the unchanged Chapter 7 implementation: 2,000 circular 60-calendar-date block resamples, seed 927601, dates synchronised across markets, minimum two-sided p=1/2,000. Per-market intervals and p-values are unadjusted and use the same panel calendar.',
        'All 646 protected original files retain their hashes. No existing scripts, results or tracked dissertation files were edited. No checkpoint was trained or changed.',
        '', 'FILES',
        'results_table.csv: final 30 panel/level/contrast rows with mean differences, intervals, raw p, primary Holm p and findings.',
        'per_market_comparisons.csv: 840 matched per-market comparisons with individual intervals and unadjusted p-values.',
        'per_market_diagnostics.csv and diagnostic_summary.csv: hit counts, rates, both hit bands and Christoffersen tests.',
        'mixtures/: all 140 regenerated seed/market mixture archives, before density combination.',
        'tails/: full per-market raw and scoring tail pairs, plus guard forecasts.',
        'matched_dates.csv, numerical_audit.csv, es_quadrature_checks.csv, backend_check.json, archived_t8_replay_check.json, preservation.json: validation and provenance.',
        '', 'No unresolved missing inputs or incomplete comparisons. Prior panel exposure, historical comparator fit acceptance and finite bootstrap resolution remain limits of the inference.'
    ]
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    deviations=dict(protocol_changes=[],unresolved_missing_inputs=[],
        declared_implementation_differences=['MPS float32 inference with sampled CPU parity check','Native guard initialises at first available forecast, index one','Historical comparator acceptance policy retained'],
        validation_observations={'maximum_archived_return_roundoff':max(x['max_return_error'] for x in replay['checks']), 'maximum_score_effect':max(x['max_score_effect_of_return_roundoff'] for x in replay['checks'])},
        reporting_note='Secondary lower/higher findings are unadjusted; the term beats is reserved for the primary Holm family in REPORT.txt and results_table.csv.',
        prior_panel_exposure=True,source_seal_unchanged=True)
    (OUT/'deviations.json').write_text(json.dumps(deviations,indent=2)+'\n')
    print(OUT/'REPORT.txt')


if __name__=='__main__':
    main()

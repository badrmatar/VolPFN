"""One post-hoc results table, with unusable model paths reported separately."""
import json,sys,platform
from pathlib import Path
import numpy,scipy,pandas,arch,numba
from run import HERE,OUT,protocol,save

NAMES={'gjr_garch_t':'GJR-GARCH-t','fhs_gjr_garch':'FHS on GJR-GARCH','es_caviar':'ES-CAViaR'}


def main():
    p=protocol();rows=json.loads((OUT/'results.json').read_text());counts=json.loads((OUT/'fit_counts.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text());failed=json.loads((OUT/'failures.json').read_text())
    lines=['POST-HOC: native VolPFN against three additional tail models','5 October 2026','',
           'Later 28-index panel, 2015–February 2022. Each complete comparison uses the same 50,073 market–date observations as the previous native-tail comparison. Native forecasts were reused unchanged.',
           'The panel and prior results were already inspected. This is an explicitly post-hoc extension, not independent confirmation.',
           'Differences are native VolPFN minus comparator FZ0: negative favours native. Means weight markets equally. Intervals are marginal 95% calendar-block intervals. Holm adjusts these nine comparisons only; previous comparisons are not pooled into this family.',
           '', 'RESULTS', 'model | level | mean difference | 95% interval | raw p | Holm p | decision']
    for r in rows:
        if r['status']=='complete':
            lines.append(f"{NAMES[r['model']]} | {100*r['alpha']:g}% | {r['mean_difference']:+.6f} | [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['raw_p']:.4f} | {r['holm_p']:.4f} | {r['decision']}")
        else:
            lines.append(f"{NAMES[r['model']]} | {100*r['alpha']:g}% | unavailable | unavailable | unavailable | unavailable | invalid paths: {r['missing_markets']}")
    lines += ['', 'VolPFN meets the frozen beats rule against GJR-GARCH-t at all three levels, FHS on GJR-GARCH at 5%, and ES-CAViaR at 1%. FHS-GJR at 2.5% does not survive Holm adjustment; FHS-GJR at 1% is a tie because its interval includes zero.',
              '', 'FITTING AND FORECAST FAILURES',
              f"GJR: {sum(x['gjr_attempts'] for x in counts.values())} scheduled fits, {sum(x['gjr_rejected'] for x in counts.values())} rejected. All GJR-t and GJR-FHS market/level paths are complete.",
              f"ES-CAViaR: 84 market/level fits and {sum(x['es_candidates'] for x in counts.values())} optimiser starts. No optimiser convergence failures were recorded. However, the selected fits generate invalid later tail forecasts in four market/level cells:"]
    for x in audit['invalid_forecast_cells']:
        lines.append(f"  {x['market']}, {100*x['alpha']:g}%: {x['invalid_scored_days']}/{x['scored_days']} invalid scored days. First {x['first_invalid_date']}: VaR={x['first_invalid_var']:+.6f}, ES={x['first_invalid_es']:+.6f}. The optimiser converged, but q>=0 and ES>VaR violate this benchmark's negative lower-tail multiplicative-ES specification.")
    lines += ['The frozen failure rule therefore withholds the complete ES-CAViaR comparisons at 5% and 2.5%. No markets were dropped, no alternate optimiser candidate was selected, and no retuning or rescue fit was attempted. Both unavailable comparisons retain p=1 solely as reserved inputs to the nine-slot Holm calculation; their reported p-values remain unavailable.',
              '', 'IMPLEMENTATIONS AND RULES',
              'GJR-GARCH-t reuses the existing Tier A implementation: constant mean, GJR(1,1), Student-t innovations, rolling 1,000 observations, refit every 20 sessions, one arch default start, maxiter=1000, existing convergence and parameter checks. This is its existing rolling procedure, not the expanding-fit schedule of the older Chapter 7 GARCH comparator.',
              'FHS uses those exact same accepted GJR-t fits and conditional variances. The empirical residual tail pair is estimated within each training window, excluding its first residual, using the existing fractional-boundary VaR/ES rule. It stays fixed between accepted refits; the conditional scale updates daily.',
              'ES-CAViaR reuses the original GP-FZ SAV recursion and multiplicative ES implementation without retuning: fit once on the first 1,000 observations, seed zero, 30 original random starts, original Nelder-Mead limits/tolerances, and the original smallest-objective selection. It is the fit-once benchmark, not the separately available rolling-refit variant.',
              'FZ0, percentage-return units, caps/floor, matched dates and the calendar bootstrap are unchanged: 2,000 circular 60-date block resamples, seed 927601, dates synchronised across markets. No scored cap activations occurred for available comparisons.',
              'Beats requires Holm p<0.05 and a negative difference. An interval containing zero takes precedence and is a tie. A negative interval without Holm significance is labelled no adjusted evidence, not beats.',
              'The inherited scoring block quotes the prior study’s policy for archived comparators; this extension’s new-model fitting is governed by the explicit GJR, FHS and ES-CAViaR blocks. No archived comparator was refitted or changed.',
              '', 'VERIFICATION AND PRESERVATION',
              'Four unit tests passed: exact compiled/uncompiled CAViaR recursion and objective parity, ES-CAViaR forecast causality, and GJR/FHS causality.',
              'The full AEX 5% ES-CAViaR fit was independently repeated through the uncompiled original recursion: all 30 candidate results and selected parameters were bit-identical. Acceleration changed execution only, with no fastmath or altered optimisation.',
              'An independent statsmodels Holm calculation agreed with the nine-test adjustment.',
              'The POST_HOC_PROTOCOL.json file and all pre-fitting sealed analysis sources are unchanged. All 716 protected prior files, including the preceding experiment, original scripts and tracked dissertation files, retain their hashes.',
              'No protocol-rule deviations, retuning, native inference reruns or changes to existing results. Unavailable ES-CAViaR paths are retained as failures rather than repaired.',
              '', 'FILES',
              'results_table.csv / results.json: the single nine-row results table.',
              'invalid_forecast_paths.csv and unavailable_models.csv: unusable market/level paths.',
              'failed_fit_attempts.csv: optimiser/refit failures (header only when none).',
              'fits/: all GJR refits, FHS residual estimates and all ES-CAViaR candidates.',
              'forecasts/: new comparator tail paths, dates and returns for available cells.',
              'audit.json, legacy_fit_parity.json, preservation.json and runtime.json: checks and provenance.',
              '', 'Saved locally. No GitHub push and no dissertation edits.']
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    save(OUT/'runtime.json',dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,arch=arch.__version__,numba=numba.__version__,workers=4))
    save(OUT/'deviations.json',dict(protocol_changes=[],retuning=False,prior_files_changed=[],unavailable_forecast_cells=audit['invalid_forecast_cells'],
                                   note='Converged ES-CAViaR fits can still generate invalid out-of-sample pairs; frozen whole-panel failure rule applied.'))
    print(OUT/'REPORT.txt')


if __name__=='__main__':main()

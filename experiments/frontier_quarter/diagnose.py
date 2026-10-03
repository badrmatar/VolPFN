"""Read-only scientific diagnostics; write a separate post-hoc report."""
import json
from pathlib import Path
import numpy as np
from experiments.frontier_benchmark.metrics import qlike
from experiments.runpod_transfer.worker import sha, save
from experiments.frontier_staged.report import validate


def load_cell(root, name):
    root = Path(root)
    study = json.loads((root/'study.json').read_text())
    if sha(root/'study.json') != (root/'study.sha256').read_text().strip():
        raise ValueError('Study hash mismatch')
    manifest = json.loads((root/'bundle/manifest.json').read_text())
    if sha(root/'bundle/manifest.json') != study['bundle_manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    cell = root/'models'/name
    status = json.loads((cell/'status.json').read_text())
    if status['status'] != 'complete' or sha(cell/'predictions.json') != status['forecast_sha256']:
        raise ValueError('Incomplete or altered forecasts')
    if name == 'tabpfn_ts' and not status.get('bridge_passed'):
        raise ValueError('Missing bridge')
    return validate(json.loads((cell/'predictions.json').read_text()), manifest['origins'])


def diagnose(root):
    root = Path(root)
    study = json.loads((root/'study.json').read_text())
    rows = {name: load_cell(root, name) for name in study['order']}
    markets = study['markets']
    result = dict(label='Post-hoc descriptive diagnostics; no row exclusions from headline',
                  inputs={name: sha(root/'models'/name/'predictions.json') for name in rows})
    losses = {}
    floors = {}
    for name, rr in rows.items():
        y = np.array([r['target'] for r in rr]); h = np.array([r['h'] for r in rr])
        losses[name] = qlike(y*y, study['scales'][name]*h)
        floors[name] = [dict(market=r['market'],date=r['date'],target=r['target'],
                            h=r['h'],squared_quantiles=r['squared_quantiles'],
                            loss=float(losses[name][i]))
                        for i,r in enumerate(rr) if r['floor_activated']]
    ref = rows['volpfn']; delta = losses['volpfn']-losses['tabpfn_ts']
    contributions = {m: float(np.mean([delta[i] for i,r in enumerate(ref) if r['market']==m]))
                     for m in markets}
    daily = []
    for i,r in enumerate(ref):
        n = sum(t['market']==r['market'] for t in ref)
        daily.append(dict(market=r['market'],date=r['date'],target=r['target'],
                          tabpfn_advantage=float(delta[i]),
                          contribution_to_headline=float(delta[i]/n/len(markets))))
    result.update(floor_events=floors,market_tabpfn_advantages=contributions,
                  tabpfn_market_wins=sum(v>0 for v in contributions.values()),
                  headline_tabpfn_advantage=float(np.mean(list(contributions.values()))),
                  leave_one_market_out={m:float(np.mean([v for k,v in contributions.items() if k!=m])) for m in markets},
                  largest_date_contributions=sorted(daily,key=lambda r:abs(r['contribution_to_headline']),reverse=True)[:15])
    save(root/'JANUARY_DIAGNOSTICS.json',result)
    lines=['# January diagnostic audit','',result['label'],'',
           f"TabPFN wins fixed-scale QLIKE in {result['tabpfn_market_wins']} of {len(markets)} markets.",
           f"Full-panel mean advantage: {result['headline_tabpfn_advantage']:.6f} QLIKE.",
           f"Descriptive omission of SSMI reverses the mean: {result['leave_one_market_out']['SSMI']:.6f}.",
           'SSMI remains in the primary table. Its omission is not a revised test.', '',
           '| Model | Positivity-floor events |','|---|---:|']
    lines += [f'| {name} | {len(events)} |' for name,events in floors.items()]
    lines += ['', 'Moirai squared-return forecasts use its median, not an expectation. Negative or',
              'near-zero median outputs are floored by the predeclared adapter. QLIKE severely',
              'penalizes a tiny variance forecast when the realised squared return is large.',
              'Inspect saved native quantiles and floor flags before interpreting this as',
              'general model failure. Neither floors nor observations are changed here.', '',
              '| Market | Date | Return | Contribution to TabPFN mean advantage |',
              '|---|---|---:|---:|']
    lines += [f"| {r['market']} | {r['date']} | {r['target']:.6f} | {r['contribution_to_headline']:.6f} |" for r in result['largest_date_contributions']]
    (root/'JANUARY_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    import sys
    r = diagnose(sys.argv[1])
    print(json.dumps({k:v for k,v in r.items() if k not in ('inputs','leave_one_market_out','market_tabpfn_advantages')},indent=2))

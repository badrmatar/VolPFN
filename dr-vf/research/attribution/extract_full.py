"""Network variance paths on the full 2000-2022 series (reserved block included) for the calibeating
study's one-shot reserved pass (calibeat-var-es/MARKET_PROTOCOL.md, rule M7).

Same checkpoint, window construction and predictive outputs as research/attribution/extract.py, but
reads research/data/oxfordman_barde/calendar_full_v1 (prepared by the calendar-preserving
prepare_market_data.py over the whole source) and does not refuse dates after 2014-12-31. This
opens reserved outcomes to the network's own forecasts only; nothing here trains, tunes, selects a
checkpoint, or scores. Run once: python3 research/attribution/extract_full.py
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from research.attribution.extract import window_owner, L_WINDOW, STRIDE, MIN_HISTORY
from research.prior_adaptation.experiment import save, sha

DATA = Path('research/data/oxfordman_barde/calendar_full_v1')
DEV = Path('research/data/oxfordman_barde/calendar_development_v2')


def run(args):
    import volpfn as V
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    hashes = {str(p): sha(p) for p in [Path(__file__), Path('volpfn.py'), Path('volpfn_prior.py'), checkpoint]}
    save(root / 'protocol.json', {'scope': 'Reserved block opened for the calibeating study (MARKET_PROTOCOL.md M7), one pass. '
                                          'Extracts the saved checkpoint own variance paths on the full series; no training, tuning, '
                                          'checkpoint selection or scoring.', 'inputs': hashes,
                                  'window': {'L': L_WINDOW, 'stride': STRIDE, 'min_history': MIN_HISTORY}, 'data': str(DATA)})
    model = V.load_model(str(checkpoint), 'cpu')
    names = sorted(p.stem for p in DEV.glob('*.npz')); started = time.perf_counter(); manifest = {}
    for name in names:
        with np.load(DATA / f'{name}.npz', allow_pickle=False) as a:
            date = a['date'].astype('U10'); r = a['returns'].astype(float) * 100.
        with np.load(DEV / f'{name}.npz', allow_pickle=False) as d:
            n = len(d['date'])
            if not (np.array_equal(d['date'].astype('U10'), date[:n]) and np.allclose(d['returns'] * 100., r[:n])):
                raise RuntimeError(f'{name}: full series does not extend the development series')
        out = V.predict_series(model, r, L=L_WINDOW, stride=STRIDE, min_history=MIN_HISTORY, levels=())
        owner = window_owner(len(r)); h = out['h']; finite = np.isfinite(h)
        if np.any(finite != (owner >= 0)):
            raise RuntimeError(f'{name}: window ownership map disagrees with which forecasts exist')
        np.savez_compressed(root / f'{name}.npz', date=date, r=r, h=h, h_lat=out['h_lat'], logscore=out['logscore'], owner=owner)
        manifest[name] = {'observations': int(len(r)), 'forecasts': int(finite.sum()), 'first_date': str(date[0]),
                          'last_date': str(date[-1]), 'development_rows': int(n)}
        print(f'{name}: {finite.sum()} forecasts, {len(r)} days to {date[-1]}', flush=True)
    save(root / 'manifest.json', {'markets': manifest, 'elapsed_seconds': time.perf_counter() - started})
    save(root / 'status.json', {'status': 'complete'}); print('COMPLETE', len(names), 'markets', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='out_volpfn/volpfn_base_best.pt')
    ap.add_argument('--out', default='research/outputs/attribution_extract_full_v1')
    run(ap.parse_args())

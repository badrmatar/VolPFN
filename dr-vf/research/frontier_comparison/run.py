"""Run a predeclared monthly development screen; never open final-period data."""
from __future__ import annotations
import os

# Set the library-specific opt-out before imports, also inherited by workers.
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/volpfn-frontier-comparison-matplotlib'

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import traceback
import numpy as np
from research.frontier_benchmark import run as R
from research.seed_replication import training as T

PACKAGE = Path(__file__).resolve().parent
PARENT = R.ROOT / 'research/outputs/frontier_tabpfn_pilot_v1'
DEFAULT_OUT = R.ROOT / 'research/outputs/frontier_development_comparison_v1'
MARKETS = ('SPX', 'FTSE', 'N225')
ORDER = ('volpfn', 'chronos2', 'moirai2', 'tabpfn_ts')
BASE_POWER_SAFE = R.power_safe


def install_power_policy(out, manifest):
    """Honor this user's explicit low-battery override in this run only."""
    authorization = manifest.get('low_battery_override', {})
    def check():
        try:
            return BASE_POWER_SAFE()
        except RuntimeError as exc:
            if (not authorization.get('authorized') or
                    not str(exc).startswith('Low discharging battery:')):
                raise
            power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
            with (Path(out)/'power_override_events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=T.now(), power=power,
                    action='continued_under_explicit_user_override'))+'\n')
            return True
    R.power_safe = check


def make_origins(panel):
    """First common observed session per month, chosen without looking at returns."""
    common = None
    for market in MARKETS:
        if market not in panel:
            raise ValueError('Missing fixed market: ' + market)
        dates = np.asarray(panel[market]['date'])
        if np.any(dates[1:] <= dates[:-1]) or np.any(dates > '2014-12-31'):
            raise ValueError('Invalid or postdevelopment dates')
        common = dates if common is None else np.intersect1d(common, dates)
    origins = []
    for year in (2012, 2013, 2014):
        phase = 'evaluation' if year == 2014 else 'calibration'
        for month in range(1, 13):
            prefix = '%04d-%02d' % (year, month)
            available = common[np.char.startswith(common.astype(str), prefix)]
            if not len(available):
                raise ValueError('No common observed session in ' + prefix)
            date = str(available[0])
            for market in MARKETS:
                index = int(np.searchsorted(panel[market]['date'], date))
                if index < 1024:
                    raise ValueError('Insufficient fixed context for ' + market)
                origins.append(dict(market=market, index=index, date=date, phase=phase))
    return origins


def freeze(out=DEFAULT_OUT):
    out = Path(out).resolve()
    parent = R.verify(PARENT)
    panel = R.locked_data(PARENT, parent)
    origins = make_origins(panel)
    if out.exists():
        raise FileExistsError('Attempts cannot be overwritten')
    out.mkdir(parents=True)
    shutil.copy2(PARENT/'development_data.npz', out/'development_data.npz')
    shutil.copytree(PARENT/'source', out/'source')
    m = copy.deepcopy(parent)
    m.update(created_utc=T.now(), kind='monthly_development_comparison_only',
        parent_pilot=str(PARENT), parent_manifest_sha256=T.sha(PARENT/'manifest.json'),
        order=list(ORDER), selected_markets=list(MARKETS), origins=origins,
        calibration_period=['2012-01-01', '2013-12-31'],
        evaluation_period=['2014-01-01', '2014-12-31'],
        timeout_seconds_per_model=7200,
        selection='Fixed three geographic indices; first common session each month; exploratory, no significance claims',
        amendment='New bounded development screen after the integration gate; fixed-final seed8101 remains the only VolPFN checkpoint; LSTM excluded pending competitive training',
        scale_calibration='Equal-market QLIKE-optimal positive factor, fit on 2012-2013 monthly origins only; applied to 2014; a screening factor, not final benchmark calibration',
        operational_environment={'TABPFN_DISABLE_TELEMETRY':'1',
            'MPLCONFIGDIR':os.environ['MPLCONFIGDIR'], 'HF_HUB_OFFLINE':'1',
            'HF_HUB_DISABLE_TELEMETRY':'1', 'DO_NOT_TRACK':'1', 'threads':'1'},
        low_battery_override={'authorized':True,
            'instruction':'run it now illll put charger in 5mins',
            'context':'User instructed immediate launch after being told battery was 8% and discharging; log low-battery events rather than block this comparison'},
        exposure='All observations are from previously inspected development years; no untouched-data or pretrained-data-independence claim')
    m['models']['lstm'] = dict(status='excluded_pending_competitive_training',
        reason='The 32-step fit verifies integration only and is not a fair trained comparator')
    for path in (PARENT/'manifest.json', PARENT/'manifest.sha256'):
        m['hashes'][str(path)] = T.sha(path)
    for path in sorted(PACKAGE.iterdir()):
        if path.suffix not in ('.py', '.md', '.command'):
            continue
        relative = str(path.relative_to(R.ROOT))
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        m['sources'][relative] = T.sha(path)
    T.save_json(out/'manifest.json', m)
    (out/'manifest.sha256').write_text(T.sha(out/'manifest.json')+'\n')
    T.save_json(out/'status.json', dict(status='frozen', created_utc=T.now(), origins=len(origins)))
    verify(out)
    return m


def verify(out):
    m = R.verify(out)
    if (m['kind'] != 'monthly_development_comparison_only' or
            m['order'] != list(ORDER) or m['selected_markets'] != list(MARKETS)):
        raise RuntimeError('Wrong comparison scope')
    if m['origins'] != make_origins(R.locked_data(out, m)):
        raise RuntimeError('Comparison origin drift')
    if os.environ.get('TABPFN_DISABLE_TELEMETRY') != '1':
        raise RuntimeError('Telemetry opt-out missing')
    return m


def worker(out, name):
    m = verify(out)
    if name not in ORDER:
        raise ValueError('Excluded comparator')
    install_power_policy(out, m)
    # The previously audited prediction path remains unchanged.
    return R.worker(out, name)


def run(out=DEFAULT_OUT):
    from .report import write_report
    out = Path(out).resolve()
    m = verify(out)
    install_power_policy(out, m)
    R.power_safe()
    with (out/'started.json').open('x') as stream:
        json.dump(dict(utc=T.now(), pid=os.getpid()), stream)
    records = []
    try:
        write_report(out)
        for name in m['order']:
            verify(out)
            R.power_safe()
            T.save_json(out/'status.json', dict(status='running', model=name,
                pid=os.getpid(), updated_utc=T.now(), completed=[r['model'] for r in records
                    if r['status']=='complete_engineering_only']))
            print('COMPARISON_START', name, 'origins', len(m['origins']), flush=True)
            status_path = out/'models'/name/'status.json'
            with (out/(name+'.log')).open('x') as log:
                try:
                    result = subprocess.run([str(R.ENVIRONMENTS[name]), '-u', '-m',
                        'research.frontier_comparison.run', 'worker', '--out', str(out),
                        '--model', name], cwd=R.ROOT, stdout=log, stderr=subprocess.STDOUT,
                        timeout=m['timeout_seconds_per_model'])
                    record = json.loads(status_path.read_text()) if status_path.exists() else dict(
                        model=name, status='failed_before_status', returncode=result.returncode)
                    if result.returncode and record['status']=='complete_engineering_only':
                        record.update(status='failed_after_predictions', returncode=result.returncode)
                    elif record['status']=='running':
                        record.update(status='failed_interrupted', returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    record = dict(model=name, status='failed_timeout',
                        seconds=m['timeout_seconds_per_model'])
            T.save_json(status_path, record)
            records.append(record)
            T.save_json(out/'records.json', records)
            write_report(out)
            print('COMPARISON_END', name, record['status'], flush=True)
        failed = [r['model'] for r in records if r['status']!='complete_engineering_only']
        T.save_json(out/'status.json', dict(status='finished' if not failed else 'finished_with_failures',
            finished_utc=T.now(), completed=[r['model'] for r in records
                if r['status']=='complete_engineering_only'], failed=failed,
            final_market_scoring_performed=False))
        write_report(out)
        return not failed
    except BaseException as exc:
        T.save_json(out/'status.json', dict(status='failed_driver', error=repr(exc), updated_utc=T.now()))
        (out/'driver_failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'verify', 'run', 'worker', 'report'))
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--model', choices=ORDER)
    a = p.parse_args()
    if a.command == 'freeze':
        m = freeze(a.out)
        print('FROZEN', a.out, 'origins', len(m['origins']))
    elif a.command == 'verify':
        verify(a.out); print('VERIFIED', a.out)
    elif a.command == 'worker':
        r = worker(a.out, a.model)
        if r['status'] != 'complete_engineering_only':
            raise SystemExit(1)
    elif a.command == 'report':
        verify(a.out)
        from .report import write_report
        write_report(a.out)
    elif not run(a.out):
        raise SystemExit(1)

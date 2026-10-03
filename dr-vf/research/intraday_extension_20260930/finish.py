"""Finite local completion driver, waiting for independently verified forecasts."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.intraday_extension_20260930.run_blends import ROOT, SEEDS, sha, write_json


OUTPUTS = ROOT / 'research/outputs'
PACKAGE = ROOT / 'research/intraday_extension_20260930'


def freeze(directory, files):
    folder = directory / 'frozen_sources'
    folder.mkdir(exist_ok=False)
    hashes = {}
    for path in files:
        path = Path(path).resolve()
        target = folder / path.name
        if target.exists():
            raise ValueError('Ambiguous frozen source basename: ' + path.name)
        shutil.copy2(path, target)
        hashes[path.name] = sha(target)
    write_json(directory / 'SOURCE_FREEZE_AMENDMENT.json', {
        'original_artifact_receipt_preserved': True, 'source_copies': hashes})


def main():
    driver = OUTPUTS / 'intraday_extension_20260930_completion_driver_v1'
    driver.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), driver / 'finish.py')
    xgb = OUTPUTS / 'intraday_extension_20260930_transfer_xgb_v1'
    print('Waiting for all-market Transfer-XGBoost independent quality gate.', flush=True)
    while True:
        quality_path = xgb / 'QUALITY_VERIFICATION.json'
        if quality_path.exists():
            try:
                quality = json.loads(quality_path.read_text())
            except json.JSONDecodeError:
                time.sleep(15)
                continue
            if quality.get('status') != 'passed':
                raise ValueError('Transfer-XGBoost did not pass independent verification')
            break
        time.sleep(15)
    print('Complete Transfer-XGBoost quality gate passed; scoring fixed roster.', flush=True)
    register = PACKAGE / 'LITERATURE_REGISTER.md'
    old_register = register.read_text()
    (driver / 'LITERATURE_REGISTER_before_completion.md').write_text(old_register)
    register.write_text(old_register.replace(
        'Standard HAR-feature branch implemented and pilot checked; the all-market five-seed run is in progress.',
        'Standard HAR-feature branch implemented; all 28 markets and five seeds are complete and independently checked.'))
    gnn = OUTPUTS / 'intraday_extension_20260930/gnnhar_v2'
    qhar = OUTPUTS / 'intraday_extension_20260930_qhar_v2'
    extras = {}
    for name, folder, seeds, quality_name, design_name, receipt_name, code, note in (
        ('gnnhar', gnn, SEEDS, 'independent_quality_review.json', 'design.json', 'receipt.json',
         'gnnhar.py', 'GNNHAR_ADAPTATION.md'),
        ('qlike_har', qhar, ('none',), 'QUALITY_VERIFICATION.json', 'DESIGN.json', 'RECEIPT.json',
         'qlike_har.py', 'QLIKE_HAR_ADAPTATION.md'),
        ('transfer_xgboost', xgb, SEEDS, 'QUALITY_VERIFICATION.json', 'DESIGN.json', 'RECEIPT.json',
         'transfer_xgb.py', 'TRANSFER_XGB_ADAPTATION.md')):
        pins = [PACKAGE / code, PACKAGE / note, folder / design_name,
                folder / receipt_name, folder / quality_name]
        if name == 'gnnhar':
            pins += sorted(folder.glob('seed_*_candidate_history.jsonl'))
            pins += sorted(folder.glob('*.pt'))
            pins += sorted(folder.glob('*.npz'))
        elif name in ('qlike_har', 'transfer_xgboost'):
            pins += sorted((folder / 'markets').glob('*_fits.json'))
        if name == 'transfer_xgboost':
            pins += [folder / 'DEPENDENCY_AUDIT.json']
        extras[name] = {'path': str(folder / 'predictions.csv'), 'seeds': seeds,
                        'quality_verification': str(folder / quality_name),
                        'source_pins': list(map(str, pins))}
    manifest = OUTPUTS / 'intraday_extension_20260930_blend_manifest_v3'
    manifest.mkdir(exist_ok=False)
    write_json(manifest / 'extras.json', extras)
    source = OUTPUTS / 'intraday_extension_20260930_comparison_final_v3'
    stats = OUTPUTS / 'intraday_extension_20260930_statistics_v1'
    grand = OUTPUTS / 'intraday_extension_20260930_grand_table_final_v1'
    report = OUTPUTS / 'intraday_extension_20260930_final_report_v1'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    stages = [
        ('comparison', 'run_blends', ['--out', str(source), '--extras-json', str(manifest / 'extras.json')],
         source, [PACKAGE / 'run_blends.py', PACKAGE / 'PROTOCOL.md', PACKAGE / 'test_blends.py']),
        ('statistics', 'statistics', ['--source', str(source), '--out', str(stats)],
         stats, []),
        ('grand_table', 'grand_table', ['--source', str(source), '--out', str(grand)],
         grand, []),  # The grand builder freezes its own code and tests.
        ('report', 'report', ['--source', str(source), '--statistics', str(stats), '--grand', str(grand),
                              '--out', str(report)],
         report, [PACKAGE / 'report.py', PACKAGE / 'PROTOCOL.md', register])]
    for label, module, args, folder, frozen in stages:
        command = [sys.executable, '-m', 'research.intraday_extension_20260930.' + module] + args
        started = time.perf_counter()
        with (driver / (label + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(driver / (label + '_execution.json'), {
            'command': command, 'exit_code': result.returncode,
            'elapsed_seconds': time.perf_counter() - started, 'output_directory': str(folder)})
        if result.returncode:
            raise RuntimeError(label + ' failed; original failed outputs/logs retained')
        if frozen:
            freeze(folder, frozen)
        if label == 'statistics':
            saved = folder / 'frozen_sources'
            saved.mkdir(exist_ok=False)
            sources = {'statistics.py': PACKAGE / 'statistics.py',
                       'test_statistics.py': PACKAGE / 'test_statistics.py',
                       'run_blends.py': PACKAGE / 'run_blends.py',
                       'combined_statistics.py': ROOT / 'research/combined_leaderboard/statistics.py',
                       'population_statistics.py': ROOT / 'research/population_transfer/statistics.py'}
            for name, path in sources.items():
                shutil.copy2(path, saved / name)
            write_json(folder / 'SOURCE_FREEZE_AMENDMENT.json', {
                'original_artifact_receipt_preserved': True,
                'copies': {name: {'source': str(p), 'sha256': sha(saved / name)}
                           for name, p in sources.items()}})
        print(label + ' complete: ' + str(folder), flush=True)
    write_json(driver / 'COMPLETION.json', {
        'status': 'outputs_complete_pending_independent_final_review',
        'comparison': str(source), 'statistics': str(stats), 'grand': str(grand), 'report': str(report),
        'new_volpfn_training': False, 'new_paid_resources': False})


if __name__ == '__main__':
    main()

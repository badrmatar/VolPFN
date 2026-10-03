"""Fixed numerical-identity audit; never computes a target loss or rank."""
from __future__ import annotations
import platform
import time
import numpy as np
import torch
from research.combined_leaderboard.common import (
    ROOT, OUTPUT, PLAN, CACHE, SEEDS, read, save, sha, now, require,
)
from research.combined_leaderboard.neural import Predictor
from research.combined_leaderboard.pipeline import verify
from .diagnose import forward, OUT


def run():
    started = time.perf_counter()
    require(not (OUT/'VALIDATION.json').exists(), 'Never overwrite an audit')
    design = read(OUT/'DIAGNOSTIC_PLAN_FREEZE.json')
    require(sha(ROOT/design['source']) == design['sha256'], 'Diagnostic design changed')
    require(design['uniform_rtol'] == 1e-5 and design['original_rtol'] == 1e-6,
            'Unexpected frozen tolerances')
    _, freeze = verify(OUTPUT)
    identity = sha(OUTPUT/'FREEZE.json')
    cells = []
    for path in sorted((OUTPUT/'cells').glob('**/forecast.json')):
        digest = sha(path)
        require(digest == (path.parent/'forecast.sha256').read_text().strip(),
                'Saved cell checksum mismatch')
        require(read(path)['freeze_sha256'] == identity, 'Saved cell belongs to another freeze')
        cells.append(dict(path=str(path.relative_to(ROOT)), sha256=digest))
    with np.load(PLAN.parent/'contexts.npz', allow_pickle=False) as z:
        contexts = np.concatenate([z['calibration'], z['evaluation']])
    plan = read(PLAN)
    origins = plan['calibration_origins'] + plan['origins']
    selected = read(OUTPUT/'pilot_neural_oldruntime_cpu_v1.json')
    require([x['seed'] for x in selected] == list(SEEDS), 'Wrong selected seed panel')
    records, manual_checks = [], []
    for seed in SEEDS:
        # The historical sampled indices address the evaluation array, not the
        # concatenated calibration+evaluation array. Reconfirm one pilot below.
        sample = next(x for x in selected if x['seed'] == seed)
        require(len(sample['indices']) == 20, 'Wrong fixed evaluation sample size')
        indices = list(range(72)) + [72+i for i in sample['indices']]
        if seed == 8101:
            indices += list(range(3712, 3776))
        require(len(set(indices)) == len(indices), 'Unexpected duplicate diagnostic cells')
        archived = np.array([r['h'] for phase in ('calibration','evaluation')
                             for r in read(CACHE/f'{seed}_{phase}.json')])
        cpu = Predictor(seed, 'cpu', 2)
        require(cpu.device.type == 'cpu', 'Only frozen CPU runtime is permitted')
        values = np.array([cpu.predict(contexts[i:i+1])[0] for i in indices])
        reference = Predictor(seed, 'cpu', 2)
        reference.model.double()
        refs = np.array([forward(reference, contexts[i], 'float64_reference') for i in indices])
        for j, i in enumerate(indices):
            cv, rv, av = float(values[j,2]), float(refs[j,2]), float(archived[i])
            require(np.isfinite([cv, rv, av]).all() and min(cv,rv,av)>0,
                    'Nonfinite/nonpositive diagnostic forecast')
            errors = dict(cpu_vs_reference=abs(cv-rv)/rv,
                          archive_vs_reference=abs(av-rv)/rv,
                          cpu_vs_archive=abs(cv-av)/av)
            records.append(dict(seed=seed,index=i,phase=origins[i]['phase'],
                market=origins[i]['market'],date=origins[i]['date'],
                cpu_moments=values[j],reference_moments=refs[j],archived_second_moment=av,
                relative_errors=errors,passed=all(e<=1e-5 for e in errors.values()),
                original_gate_passed=errors['cpu_vs_archive']<=1e-6))
        # This verifies evaluation-vs-concatenated indexing against the exact
        # previously saved pilot error, without reading any target or loss.
        first = records[-len(indices)+72]['relative_errors']['cpu_vs_archive']
        require(first == sample['per_origin_relative_error'][0], 'Pilot sample identity differs')
        reference.model.use_sdpa = False
        for i in (0,71):
            manual = np.asarray(forward(reference,contexts[i],'float64_reference'))
            sdpa = refs[indices.index(i)]
            passed = bool(np.allclose(manual,sdpa,rtol=1e-12,atol=1e-12))
            manual_checks.append(dict(seed=seed,index=i,manual_moments=manual,
                sdpa_moments=sdpa,absolute_error=np.abs(manual-sdpa),passed=passed,
                rtol=1e-12,atol=1e-12))
        print('VALIDATED',seed,len(indices),'cells',flush=True)
    require(len(records)==524 and len(manual_checks)==10,'Diagnostic panel incomplete')
    # Revalidate unchanged source/input/cell identities after execution.
    verify(OUTPUT)
    require(all(sha(ROOT/c['path'])==c['sha256'] for c in cells), 'Old cell changed during audit')
    result = dict(created_utc=now(),passed=all(r['passed'] for r in records)
        and all(r['passed'] for r in manual_checks),records=records,manual_checks=manual_checks,
        checked=len(records),manual_checked=len(manual_checks),
        maximum_relative_errors={k:max(r['relative_errors'][k] for r in records)
            for k in ('cpu_vs_reference','archive_vs_reference','cpu_vs_archive')},
        original_gate_exceedances=sum(not r['original_gate_passed'] for r in records),
        original_gate_exceedance_rows=[{k:r[k] for k in ('seed','index','market','date')}
            for r in records if not r['original_gate_passed']],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                     device='cpu',threads=2,batch_size=1),
        freeze_sha256=identity,diagnostic_plan_freeze_sha256=sha(OUT/'DIAGNOSTIC_PLAN_FREEZE.json'),
        validation_code_sha256=sha(__file__),reference_helper_sha256=sha(ROOT/'research/combined_leaderboard_repair/diagnose.py'),
        old_cells=cells,elapsed_seconds=time.perf_counter()-started,
        scope='Fixed numerical diagnostics only; no new forecast losses, calibration, rankings or inference inspected')
    save(OUT/'VALIDATION.json',result)
    require(result['passed'],'Fixed diagnostic gate failed; do not activate amendment')
    print('PASS',result['maximum_relative_errors'],flush=True)
    return result


if __name__ == '__main__':
    run()

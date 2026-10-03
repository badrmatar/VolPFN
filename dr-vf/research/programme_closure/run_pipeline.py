"""Execute the finite approved sequence; stop computation at the receipt deadline.

This worker timeout does NOT stop RunPod billing. The operator must stop the pod
via its control plane after export or at the recorded cloud-stop deadline.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def now(): return datetime.now(timezone.utc)


def run(root, authorization, evaluation_plan, confirmation_plan, revision='v1'):
    root=Path(root).resolve();receipt=Path(authorization).resolve()
    out=root/'research/outputs/programme_closure_20260919_v1'
    a=json.loads(receipt.read_text())
    deadline=datetime.fromisoformat(a['stop_deadline_utc'].replace('Z','+00:00'))
    if not a.get('approved_full_battery') or deadline<=now():
        raise ValueError('No current authorization')
    if revision not in ('v1','gjr-v2'):
        raise ValueError('Unknown fixed programme revision')
    suffix='' if revision=='v1' else '_v2'
    marker=out/('PIPELINE_STARTED'+suffix+'.json')
    with marker.open('x') as stream:
        json.dump(dict(started_utc=now().isoformat(),authorization=str(receipt),
                       limitation='Worker deadline only; RunPod stop still requires control-plane operator'),stream,indent=2)
    base=[sys.executable,'-m']
    design=out/('gpu_design_v1' if revision=='v1' else 'gpu_design_v2')
    training=out/('gpu_battery_v1' if revision=='v1' else 'gpu_battery_v2')
    training_module='research.programme_closure.gpu_training' if revision=='v1' else 'research.programme_closure.gjr_recovery'
    training_action='full' if revision=='v1' else 'run'
    commands=[
        ('training',base+[training_module,training_action,'--frozen',str(design),
            '--out',str(training),'--authorization',str(receipt)]),
        ('matched_primary',base+['research.programme_closure.evaluate','run','--plan',str(evaluation_plan),
            '--training',str(training),'--data-lock',str(out/'later_index_data_v1/manifest.json'),
            '--out',str(out/('matched_primary_v1' if revision=='v1' else 'matched_primary_v2'))]),
        ('procedure_confirmation',base+['research.programme_closure.confirm_benchmark','run',
            '--plan',str(confirmation_plan),'--training-root',str(training),
            '--out',str(out/('benchmark_confirmation_v1' if revision=='v1' else 'benchmark_confirmation_v2'))])]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
    records=[]
    for phase,command in commands:
        if now()>=deadline:
            (out/('PIPELINE_STATUS'+suffix+'.json')).write_text(json.dumps(dict(status='incomplete_deadline',records=records),indent=2)+'\n')
            raise RuntimeError('Deadline reached before next phase; operator must stop cloud pod')
        record=dict(phase=phase,started_utc=now().isoformat(),status='running')
        with (out/(phase+suffix+'.log')).open('x') as logfile:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=max(0,(deadline-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                record.update(status='worker_deadline_reached',ended_utc=now().isoformat())
                records.append(record)
                (out/('PIPELINE_STATUS'+suffix+'.json')).write_text(json.dumps(dict(status='incomplete_deadline',records=records),indent=2)+'\n')
                raise RuntimeError('Worker deadline reached; operator must stop cloud pod immediately')
        record.update(status='complete' if code==0 else 'failed',exit_code=code,ended_utc=now().isoformat())
        records.append(record)
        (out/('PIPELINE_STATUS'+suffix+'.json')).write_text(json.dumps(dict(status='running' if code==0 else 'failed',records=records),indent=2)+'\n')
        if code!=0: raise RuntimeError('Programme stage failed: '+phase)
    (out/('PIPELINE_STATUS'+suffix+'.json')).write_text(json.dumps(dict(status='complete_export_and_cloud_stop_required',records=records),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--authorization',required=True,type=Path)
    p.add_argument('--evaluation-plan',required=True,type=Path)
    p.add_argument('--confirmation-plan',required=True,type=Path)
    p.add_argument('--revision',choices=['v1','gjr-v2'],default='v1')
    a=p.parse_args();run(a.root,a.authorization,a.evaluation_plan,a.confirmation_plan,a.revision)

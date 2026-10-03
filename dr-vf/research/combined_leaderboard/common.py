from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'research/outputs/combined_leaderboard_v1'
OLD = ROOT / 'research/outputs/programme_closure_20260919_v1'
PLAN = OLD / 'benchmark_confirmation_plan_v3/manifest.json'
CACHE = OLD / 'remote_export_v2/benchmark_confirmation_resume_v1'
FRONTIER = ROOT / 'research/outputs/frontier_overnight_increment_201504_201606_v1'
DEV = ROOT / 'research/outputs/frontier_development_comparison_cuda_v1'
PREP = ROOT / 'research/outputs/tier_a_confirmation_prepare_v2/PREPARATION.json'
SEEDS = (8101, 8102, 8103, 8104, 8105)
FMS = ('chronos2', 'moirai2', 'tabpfn_ts')

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def save(path, x, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if replace else 'x'
    with path.open(mode) as f:
        json.dump(x, f, indent=2, allow_nan=False, default=default)
        f.write('\n')

def key(row):
    return row['market'], row['date'], row['phase']

def require(ok, message):
    if not ok: raise ValueError(message)

def load_bundle(out):
    p = Path(out) / 'bundle.json'
    b = read(p)
    for name, digest in b['inputs'].items():
        require(sha(ROOT/name) == digest, 'Input hash changed: '+name)
    return b

def market_data(bundle, name):
    path = ROOT / bundle['market_paths'][name]
    with np.load(path, allow_pickle=False) as z:
        keep = z['date'].astype('U10') <= '2016-06-30'
        return dict(name=name, **{k:z[k][keep].copy() for k in ('date','r','rk')})

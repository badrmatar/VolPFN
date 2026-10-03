"""Unchanged corrected weights on independently reset 1,024-return contexts.

Retains the archived mixed-float32 second-moment arithmetic, adding the mean
needed for a centred-variance score. No training, quantile interpolation, or
reuse of the different anchored-window Tier A predictions occurs here.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from research.return_only_benchmark.volpfn_adapter import configure_device, pinned_checkpoint, _synchronize
from research.seed_replication import causal
from .common import require

CONTEXT = 1024
BRIDGE_RTOL = 1e-6

class Predictor:
    def __init__(self, seed, device='cpu', threads=2):
        self.device = configure_device(device, threads)
        path, self.entry = pinned_checkpoint(seed)
        self.model = causal.load_model(str(path), self.device)

    @torch.inference_mode()
    def predict(self, contexts):
        contexts = np.asarray(contexts, dtype=np.float64)
        require(contexts.ndim == 2 and contexts.shape[1] == CONTEXT and len(contexts)>0,
                'Expected nonempty independent N x 1024 contexts')
        features = causal.feature_batch(contexts)
        prediction = self.model(torch.from_numpy(features['feats']).to(self.device),
                                torch.from_numpy(features['base']).to(self.device))
        last = {k:prediction[k][:,-1].float().cpu().numpy()
                for k in ('logw','loc','logscale','nu')}
        values = []
        # Deliberately use the historical per-row NumPy arithmetic. Promoting
        # the whole reduction to float64 changes its last bits against the cache.
        for i in range(len(contexts)):
            unit = float(features['output_scale'][i,-1])
            w, nu = np.exp(last['logw'][i]), last['nu'][i]
            loc = last['loc'][i] * unit
            scale = np.exp(last['logscale'][i]) * unit
            require(np.all(nu>2) and np.all(scale>0) and np.isfinite([w,nu,loc,scale]).all(),
                    'Invalid native mixture parameters')
            require(np.isclose(w.sum(),1,rtol=2e-6,atol=2e-7), 'Invalid mixture weights')
            second = float(np.sum(w*(loc**2+scale**2*nu/(nu-2))))
            mean = float(np.sum(w*loc))
            variance = second-mean**2
            require(np.isfinite([mean,variance,second]).all() and variance>0,
                    'Invalid native moments; no forecast substitution')
            values.append((mean,variance,second))
        return np.array(values)

def compare(moment, archived):
    moment, archived = np.asarray(moment,float), np.asarray(archived,float)
    require(moment.shape==archived.shape and np.all(archived>0), 'Invalid bridge inputs')
    relative = np.abs(moment-archived)/archived
    return dict(passed=bool(np.isfinite(relative).all() and np.all(relative<=BRIDGE_RTOL)),
                checked=int(relative.size), maximum_relative_error=float(relative.max()),
                rtol=BRIDGE_RTOL, atol=0.)

def pilot(seed, contexts, archived, device='cpu', threads=2, batch_size=1):
    started=time.perf_counter()
    predictor=Predictor(seed,device,threads)
    predictor.predict(contexts[:1]); _synchronize(device)
    warmup=time.perf_counter()-started
    start=time.perf_counter()
    predictions=np.concatenate([predictor.predict(contexts[i:i+batch_size])
                               for i in range(0,len(contexts),batch_size)])
    _synchronize(device)
    elapsed=time.perf_counter()-start
    return dict(seed=seed,device=device,threads=threads,batch_size=batch_size,
                forecasts=len(contexts),warmup_seconds=warmup,elapsed_seconds=elapsed,
                seconds_per_forecast=elapsed/len(contexts),
                projected_44230_forecast_hours=elapsed/len(contexts)*44230/3600,
                bridge=compare(predictions[:,2],archived),
                scope='Calibration-only engineering timing; no new market loss inspected')

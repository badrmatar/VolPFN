"""Same causal inputs and mixture parameterization, learned from real returns."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import volpfn as V
from research.seed_replication import causal,training as T
from . import data,metrics


class PooledLSTM(nn.Module):
    def __init__(self,width=544,layers=2,K=4):
        super().__init__()
        self.cfg=dict(width=width,layers=layers,K=K)
        self.inp=nn.Linear(8,width)
        self.rnn=nn.LSTM(width,width,num_layers=layers,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,4*K)
        nn.init.normal_(self.head.weight,std=.02)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad(): self.head.bias[3*K:]=1.4

    def forward(self,feats,base):
        x,_=self.rnn(self.inp(feats))
        o=self.head(self.norm(x));K=self.cfg['K']
        logits,a,b,c=o[...,:K],o[...,K:2*K],o[...,2*K:3*K],o[...,3*K:]
        return dict(logw=torch.log_softmax(logits,-1),loc=torch.exp(.5*base)[...,None]*a,
            logscale=.5*base[...,None]+b.clamp(-6,6),nu=2.1+torch.exp(c.clamp(-4,6)))


def real_batch(histories,rng,batch_size,length):
    names=sorted(histories)
    selected=rng.choice(names,size=batch_size)
    returns=[]
    for market in selected:
        r=histories[market]['r']
        if len(r)<length+1: raise ValueError('Fit history too short')
        start=int(rng.integers(0,len(r)-length))
        returns.append(r[start:start+length+1])
    raw=np.asarray(returns)
    f=causal.feature_batch(raw[:,:-1])
    return f,raw[:,1:]/f['output_scale']


def fit(held_out,out,steps=32,width=544,length=128,batch_size=2,seed=91801,histories=None):
    """Bounded development engineering fit, never labelled full comparator."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    histories=data.training_histories(held_out) if histories is None else histories
    if held_out in histories or not histories or any(
            not len(v['date']) or np.any(v['date']>'2010-12-31') or
            len(v['date'])!=len(v['r']) for v in histories.values()):
        raise ValueError('Held-out identity or future date entered real-data training')
    model=PooledLSTM(width=width)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.98),weight_decay=.01)
    record=dict(status='running',kind='bounded_engineering_only',held_out=held_out,
        fit_markets=sorted(histories),fit_end='2010-12-31',steps=steps,cfg=model.cfg,
        batch_size=batch_size,context_length=length,seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),loss='raw return NLL; no unavailable latent supervision')
    T.save_json(out/'status.json',record)
    started=time.perf_counter();losses=[]
    for step in range(steps):
        batch,y=real_batch(histories,rng,batch_size,length)
        pred=model(torch.from_numpy(batch['feats']),torch.from_numpy(batch['base']))
        target=torch.tensor(y,dtype=torch.float32)
        loss=(-V.mixture_logpdf(pred,target)+torch.tensor(batch['output_log_scale'])).mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite LSTM loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();T.finite_parameters(model);losses.append(float(loss.detach()))
        if (step+1)%8==0: print('LSTM_PILOT',step+1,losses[-1],flush=True)
    torch.save(dict(state_dict=model.state_dict(),cfg=model.cfg,record=record),out/'final.pt')
    record.update(status='complete_engineering_only',elapsed_seconds=time.perf_counter()-started,
        loss_first=losses[0],loss_last=losses[-1],checkpoint_sha256=T.sha(out/'final.pt'))
    T.save_json(out/'status.json',record)
    return model.eval(),record


@torch.no_grad()
def predict(model,past):
    f=causal.feature_batch(np.asarray(past,float)[None])
    p=model(torch.from_numpy(f['feats']),torch.from_numpy(f['base']))
    scale=float(f['output_scale'][0,-1])
    logw=p['logw'][0,-1].numpy();loc=p['loc'][0,-1].numpy()*scale
    scales=np.exp(p['logscale'][0,-1].numpy())*scale;nu=p['nu'][0,-1].numpy()
    h=float(np.sum(np.exp(logw)*(loc**2+scales**2*nu/(nu-2))))
    return dict(h=h,quantiles=metrics.mixture_quantiles(logw,loc,scales,nu),
        logw=logw,loc=loc,scale=scales,nu=nu)

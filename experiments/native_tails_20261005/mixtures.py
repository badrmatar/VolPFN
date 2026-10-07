"""Expose raw mixture parameters using the frozen corrected window ownership."""
import numpy as np
import torch
from volpfn.preprocessing import feature_batch


@torch.inference_mode()
def predict_mixture(model, returns, L=1024, stride=512, batch_windows=4, min_history=1):
    r = np.asarray(returns,dtype=np.float64)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError('Finite one-dimensional returns required')
    if not 1 <= stride <= L//2 or batch_windows < 1 or min_history < 1:
        raise ValueError('Invalid window settings')
    model.eval()
    device = next(model.parameters()).device
    size = (len(r),model.cfg['K'])
    result = {key:np.full(size,np.nan) for key in ('weight','location','scale','df')}
    starts = [0]+list(range(stride,max(len(r)-stride,0),stride))
    windows=[]
    for k,start in enumerate(starts):
        seg=r[start:start+L]
        first=max(1 if k==0 else stride,min_history-start)
        if len(seg)>first:
            windows.append((start,seg,first))
    i=0
    while i<len(windows):
        group=[windows[i]];i+=1
        while i<len(windows) and len(windows[i][1])==len(group[0][1]) and len(group)<batch_windows:
            group.append(windows[i]);i+=1
        features=feature_batch(np.stack([g[1] for g in group]))
        output=model(torch.as_tensor(features['feats'],device=device),torch.as_tensor(features['base'],device=device))
        out={k:v.float().cpu().numpy() for k,v in output.items()}
        for j,(start,seg,first) in enumerate(group):
            pos=np.arange(first-1,len(seg)-1);target=start+pos+1
            unit=features['output_scale'][j,pos]
            # Match the original inference arithmetic, then normalise weights in float64.
            logscale=out['logscale'][j,pos]+np.log(unit)[:,None]
            weights=np.exp(out['logw'][j,pos].astype(np.float64))
            result['weight'][target]=weights/weights.sum(axis=1,keepdims=True)
            result['location'][target]=out['loc'][j,pos]*unit[:,None]
            result['scale'][target]=np.exp(logscale)
            result['df'][target]=out['nu'][j,pos]
    return result


def combine(seeds):
    if len(seeds)!=5:
        raise ValueError('All five seed densities are required')
    return {key:np.concatenate([s[key]/5 if key=='weight' else s[key] for s in seeds],axis=1)
            for key in ('weight','location','scale','df')}

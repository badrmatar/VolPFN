"""Predictive and auxiliary losses with their original target alignment."""
import torch
from .distributions import mixture_logpdf

def legacy_losses(out, r, logv, mask):
    """next-step losses: out at positions 0..T-2 predicts r[1:], logv[1:]; mask[1:] weights the targets."""
    nll = -mixture_logpdf({k: v[:, :-1] for k, v in out.items()}, r[:, 1:])
    m = out['lat_mean'][:, :-1]
    ls = out['lat_logsd'][:, :-1]
    aux = 0.5 * ((logv[:, 1:] - m) * torch.exp(-ls)) ** 2 + ls
    w = mask[:, 1:]
    den = w.sum().clamp_min(1.0)
    return (nll * w).sum() / den, (aux * w).sum() / den

def losses(out, batch):
    """Mean raw-unit predictive NLL and latent log-variance auxiliary loss."""
    prediction = {k: v[:, :-1] for k, v in out.items()}
    nll = -mixture_logpdf(prediction, batch['target_r']) + batch['output_log_scale']
    m, ls = prediction['lat_mean'], prediction['lat_logsd']
    aux = 0.5 * ((batch['target_logv'] - m) * torch.exp(-ls)) ** 2 + ls
    w = batch['target_mask']
    den = w.sum().clamp_min(1.0)
    return (nll * w).sum() / den, (aux * w).sum() / den

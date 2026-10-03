"""Student-t mixture densities, moments and tail summaries."""
import math
import numpy as np
import torch
from scipy.special import stdtr

def lgamma_safe(x):
    try:
        return torch.lgamma(x)
    except Exception:  # Stirling with a two-step shift (error < 1e-6 for x >= 1)
        y = x + 2.0
        return (
            (y - 0.5) * torch.log(y)
            - y
            + 0.5 * math.log(2 * math.pi)
            + 1 / (12 * y)
            - 1 / (360 * y**3)
            - torch.log(x * (x + 1))
        )


def t_logpdf_torch(z, nu):
    return (
        lgamma_safe((nu + 1) / 2)
        - lgamma_safe(nu / 2)
        - 0.5 * torch.log(nu * math.pi)
        - 0.5 * (nu + 1) * torch.log1p(z * z / nu)
    )


def mixture_logpdf(out, r):
    """log predictive density of r (B,T) under the mixture at the same positions."""
    z = (r[..., None] - out['loc']) / torch.exp(out['logscale'])
    return torch.logsumexp(out['logw'] + t_logpdf_torch(z, out['nu']) - out['logscale'], dim=-1)


def mixture_second_moment(out):
    nu = out['nu']
    return torch.sum(
        torch.exp(out['logw']) * (torch.exp(2 * out['logscale']) * nu / (nu - 2) + out['loc'] ** 2),
        dim=-1,
    )


def t_logpdf_np(z, nu):
    from scipy.special import gammaln

    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - 0.5 * (nu + 1) * np.log1p(z * z / nu)
    )


def mixture_var_es(w, loc, scale, nu, level, iters=60):
    """VaR and ES (positive losses) of t-mixtures, vectorised over rows: w, loc, scale, nu of shape (T, K)."""
    a = 1 - level
    lo = np.nanmin(loc - 60 * scale, axis=1)
    hi = np.nanmax(loc + 60 * scale, axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = np.sum(w * stdtr(nu, (mid[:, None] - loc) / scale), axis=1)
        lo = np.where(cdf < a, mid, lo)
        hi = np.where(cdf < a, hi, mid)
    q = 0.5 * (lo + hi)
    z = (q[:, None] - loc) / scale
    partial = loc * stdtr(nu, z) - scale * (nu + z * z) / (nu - 1) * np.exp(t_logpdf_np(z, nu))
    es = np.sum(w * partial, axis=1) / a
    return -q, -es


def summarise(r, logw, loc, scale, nu, lat_m, lat_s, levels):
    T = len(r)
    ok = ~np.isnan(logw[:, 0])
    w = np.exp(logw)
    out = dict(
        logw=logw,
        loc=loc,
        scale=scale,
        nu=nu,
        h=np.full(T, np.nan),
        h_lat=np.full(T, np.nan),
        logscore=np.full(T, np.nan),
        mean=np.full(T, np.nan),
    )
    out['h'][ok] = np.sum(w[ok] * (scale[ok] ** 2 * nu[ok] / (nu[ok] - 2) + loc[ok] ** 2), axis=1)
    out['mean'][ok] = np.sum(w[ok] * loc[ok], axis=1)
    out['h_lat'][ok] = np.exp(lat_m[ok] + 0.5 * lat_s[ok] ** 2)
    z = (r[ok, None] - loc[ok]) / scale[ok]
    from scipy.special import logsumexp

    out['logscore'][ok] = logsumexp(logw[ok] + t_logpdf_np(z, nu[ok]) - np.log(scale[ok]), axis=1)
    for lv in levels:
        q, es = mixture_var_es(w[ok], loc[ok], scale[ok], nu[ok], lv)
        out[f'var{lv:g}'] = np.full(T, np.nan)
        out[f'es{lv:g}'] = np.full(T, np.nan)
        out[f'var{lv:g}'][ok] = q
        out[f'es{lv:g}'][ok] = es
    return out

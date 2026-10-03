import numpy as np
from scipy.special import stdtr

QUANTILES=np.arange(1,10,dtype=float)/10


def pinball(target,quantiles,levels=QUANTILES):
    y=np.asarray(target,float)
    q=np.asarray(quantiles,float)
    if q.shape!=y.shape+(len(levels),) or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError('Invalid quantile/target shape or values')
    e=y[...,None]-q
    return np.maximum(np.asarray(levels)*e,(np.asarray(levels)-1)*e).mean(-1)


def positive_point(point,past):
    if not np.isfinite(point):
        raise ValueError('Nonfinite point; no fallback')
    floor=max(1e-12,1e-8*float(np.mean(np.asarray(past,float)**2)))
    return max(float(point),floor),bool(point<floor),floor


def qlike(y,h):
    y,h=np.asarray(y,float),np.asarray(h,float)
    if np.any(y<0) or np.any(h<=0) or not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError('Invalid QLIKE arguments')
    return y/h+np.log(h)


def fit_scale(y,h,market,date):
    """Closed-form equal-market QLIKE scale; no postdevelopment targets."""
    y,h=np.asarray(y,float),np.asarray(h,float)
    market,date=np.asarray(market),np.asarray(date)
    if not (y.shape==h.shape==market.shape==date.shape) or y.ndim!=1 or not len(y):
        raise ValueError('Mismatched calibration rows')
    if np.any(date<'2011-01-01') or np.any(date>'2014-12-31'):
        raise ValueError('Scale calibration outside fixed development dates')
    qlike(y,h)
    scale=float(np.mean([np.mean(y[market==m]/h[market==m]) for m in np.unique(market)]))
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Degenerate calibration, no default substitution')
    return scale


def mixture_quantiles(logw,loc,scale,nu,levels=QUANTILES):
    logw,loc,scale,nu=map(lambda a:np.asarray(a,float),(logw,loc,scale,nu))
    if not (logw.shape==loc.shape==scale.shape==nu.shape) or np.any(scale<=0) or np.any(nu<=2):
        raise ValueError('Invalid Student mixture')
    w=np.exp(logw)
    if not np.isfinite([logw,loc,scale,nu]).all() or not np.isclose(w.sum(),1.):
        raise ValueError('Invalid mixture weights/values')
    result=[]
    for p in levels:
        lo=float(np.min(loc-scale*8));hi=float(np.max(loc+scale*8))
        def cdf(x): return float(np.sum(w*stdtr(nu,(x-loc)/scale)))
        for _ in range(60):
            if cdf(lo)<=p<=cdf(hi): break
            span=hi-lo;lo-=span;hi+=span
        else: raise FloatingPointError('Quantile bracket failed')
        for _ in range(70):
            mid=(lo+hi)/2
            if cdf(mid)<p: lo=mid
            else: hi=mid
        result.append((lo+hi)/2)
    return np.array(result)

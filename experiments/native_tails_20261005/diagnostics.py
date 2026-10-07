"""Unadjusted coverage diagnostics and the six-test Holm adjustment."""
import numpy as np
from scipy import special, stats


def holm(pvalues):
    p=np.asarray(pvalues,float)
    order=np.argsort(p,kind='stable')
    adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty_like(p);out[order]=adjusted
    return out


def coverage(hits,alpha):
    h=np.asarray(hits,dtype=int)
    n=len(h);x=int(h.sum())
    def binomial_ll(k,total,p):
        return special.xlogy(k,p)+special.xlogy(total-k,1-p)
    uc=max(0.,2*(binomial_ll(x,n,x/n)-binomial_ll(x,n,alpha)))
    counts=np.bincount(h[:-1]*2+h[1:],minlength=4).reshape(2,2)
    row=counts.sum(axis=1);success=counts[:,1]
    rates=np.divide(success,row,out=np.zeros(2),where=row>0)
    common=success.sum()/row.sum()
    ind=max(0.,2*(sum(binomial_ll(k,total,p) for k,total,p in zip(success,row,rates))
                    -sum(binomial_ll(k,total,common) for k,total in zip(success,row))))
    return dict(hits=x,days=n,hit_rate=x/n,lr_uc=float(uc),lr_ind=float(ind),
                p_ind=float(stats.chi2.sf(ind,1)),p_cc=float(stats.chi2.sf(uc+ind,2)),
                transitions=counts.tolist())

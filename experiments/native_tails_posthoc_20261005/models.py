"""Adapters for the frozen post-hoc comparators; original files stay untouched."""
from pathlib import Path
import sys
import warnings
from unittest.mock import patch
import numpy as np
from numba import njit
from experiments.tier_a_confirmation import econometric as E
from calibeat.fz import empirical_var_es, student_t_var_es

sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy_gp_fz'))
import benchmark_es_caviar as ES

ORIGINAL_PATH=ES.caviar_path
COMPILED_PATH=njit(cache=False,fastmath=False)(ORIGINAL_PATH)


def clean_float(x):
    x=float(x)
    return x if np.isfinite(x) else None


def gjr_paths(r,dates,alphas):
    """Reuse the existing GJR fit/filter; collect residual tails at accepted fits."""
    accepted={};fits=[];fhs_errors=[]
    original_fit=E._fit_one
    def record_fit(history,model_name,history_start,origin):
        state,attempt=original_fit(history,model_name,history_start,origin)
        if state is not None:
            try:
                residuals=history-state.parameters['mu']
                h=np.empty(len(history))
                state.volatility.compute_variance(state.volatility_parameters,residuals,h,
                                                  state.backcast,state.training_bounds)
                np.testing.assert_allclose(h[-1],attempt['last_training_variance'],rtol=1e-9,atol=1e-10)
                if not np.isfinite(h).all() or np.any(h<=0):raise ValueError('Invalid GJR training variance')
                z=residuals[1:]/np.sqrt(h[1:])
                pairs={str(a):empirical_var_es(z,a) for a in alphas}
                if not np.isfinite(list(pairs.values())).all():raise ValueError('Nonfinite FHS residual tail')
                accepted[str(dates[origin])]=pairs
                fits.append(dict(origin=str(dates[origin]),history_start=str(dates[history_start]),
                                 history_end=str(dates[origin-1]),residual_count=len(z),residual_pairs=pairs))
            except Exception as exc:
                fhs_errors.append(dict(origin=str(dates[origin]),reason=f'{type(exc).__name__}: {exc}'))
        return state,attempt
    with patch.object(E,'_fit_one',record_fit):
        path=E.forecast_econometric(dict(r=r,date=dates,return_units='percent'),'gjr_garch_t',
                                    start_date='2015-01-01',history_window=1000,refit_every=20)
    scored=dates>='2015-01-01'
    results={};states={}
    for a in alphas:
        if path['status']!='complete':
            states[f'gjr_garch_t|{a}']=dict(status='unavailable',reason=path.get('unavailable_reason','incomplete'))
            states[f'fhs_gjr_garch|{a}']=dict(status='unavailable',reason=path.get('unavailable_reason','incomplete'))
            continue
        nu=path['nu'][scored];mu=path['mean'][scored];sigma=np.sqrt(path['variance'][scored])
        # Existing scalar helper is evaluated at the fitted degrees of freedom for each date.
        from scipy import stats
        quant=stats.t.ppf(a,nu)
        unit=np.sqrt((nu-2)/nu)
        v=mu+sigma*unit*quant
        es=mu-sigma*unit*(nu+quant*quant)/(nu-1)*stats.t.pdf(quant,nu)/a
        results[f'gjr_garch_t|{a}']=(v,es)
        states[f'gjr_garch_t|{a}']=dict(status='complete')
        if fhs_errors:
            states[f'fhs_gjr_garch|{a}']=dict(status='unavailable',reason='accepted_fit_residual_tail_failed')
        else:
            empirical=np.asarray([accepted[o][str(a)] for o in path['fit_origin'][scored]])
            results[f'fhs_gjr_garch|{a}']=(mu+sigma*empirical[:,0],mu+sigma*empirical[:,1])
            states[f'fhs_gjr_garch|{a}']=dict(status='complete')
    audit=dict(gjr_status=path['status'],metadata=path['metadata'],attempts=path['attempts'],
               failures=path['failures'],fhs_fits=fits,fhs_errors=fhs_errors)
    return results,states,audit


def es_caviar_path(r,alpha,*,compiled=True):
    """Keep the old fit and strict-minimum selection; only observe optimiser diagnostics."""
    candidates=[];original_minimize=ES.minimize
    best_val=np.inf;best_index=None
    def record_minimize(*args,**kwargs):
        nonlocal best_val,best_index
        result=original_minimize(*args,**kwargs)
        index=len(candidates)
        candidates.append(dict(start=index,success=bool(result.success),message=str(result.message),
                               objective=clean_float(result.fun),nit=int(result.nit),nfev=int(result.nfev),
                               parameters=[clean_float(v) for v in result.x]))
        if result.fun<best_val:best_val=result.fun;best_index=index
        return result
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        with patch.object(ES,'minimize',record_minimize),patch.object(ES,'caviar_path',COMPILED_PATH if compiled else ORIGINAL_PATH):
            params,q0=ES.fit_es_caviar(r[:1000],alpha,seed=0)
            audit=dict(alpha=alpha,fit_n=1000,seed=0,candidates=candidates,selected_start=best_index,
                       q0=clean_float(q0),status='unavailable')
            if best_index is None or params is None:
                audit['reason']='no_selected_candidate';return None,audit
            selected=candidates[best_index]
            if not selected['success']:
                audit['reason']='selected_candidate_did_not_converge';return None,audit
            if not np.isfinite(params).all() or not np.isfinite(best_val) or best_val>=1e10:
                audit['reason']='invalid_selected_parameters_or_objective';return None,audit
            q=ES.caviar_path(r,*params[:3],q0)
            es=(1+np.exp(params[3]))*q
    audit['warnings']=[str(x.message) for x in captured]
    audit['parameters']=list(map(float,params))
    audit['status']='complete'
    return (q,es),audit

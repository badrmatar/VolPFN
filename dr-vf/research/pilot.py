"""Exploratory equal-training-time experiment. Not a confirmatory publication run.

Conditional targets average M fresh Gaussian draws given generating parameter and
past returns. In this fixed-unit simulator, iterated expectation preserves NLL.
This is finite Monte Carlo conditional training, NOT exact integration. Auxiliary
loss disabled in BOTH arms to isolate the predictive objective.
"""
import argparse,copy,hashlib,json,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
import torch
import volpfn as V
from research.grid_reference import simulate,reference,parameter_grid


def batch_tensors(r):
    f,b=V.make_features(r)
    return torch.tensor(f),torch.tensor(b)


def conditional_loss(out,x):
    z=(x[...,None]-out['loc'][:,:,None,:])/out['logscale'][:,:,None,:].exp()
    lp=torch.logsumexp(out['logw'][:,:,None,:]+V.t_logpdf_torch(z,out['nu'][:,:,None,:])-out['logscale'][:,:,None,:],-1)
    return -lp.mean()


def make_eval(seed,B,T,heldout=False):
    r,h,ids=simulate(seed,B,T,parameter_grid(heldout))
    ref=reference(r)
    return dict(r=r,h=h,ids=ids,reference=ref,features=batch_tensors(r))


@torch.no_grad()
def evaluate(model,data,burn):
    model.eval(); out=model(*data['features'])
    o={k:v[:,:-1] for k,v in out.items()}
    ld=V.mixture_logpdf(o,torch.tensor(data['r'][:,1:],dtype=torch.float32)).numpy()
    # output index 0 forecasts target time 1; include targets at t >= burn.
    model_path=-ld[:,burn-1:].mean(1)
    reference_path=-data['reference']['logdensity'][:,burn:].mean(1)
    oracle_path=(.5*(np.log(2*np.pi*data['h'][:,burn:])+data['r'][:,burn:]**2/data['h'][:,burn:])).mean(1)
    gap=model_path-reference_path
    model.train()
    return {'nll':float(model_path.mean()),'reference_nll':float(reference_path.mean()),'known_parameter_nll':float(oracle_path.mean()),
            'gap_to_grid_reference':float(gap.mean()),'gap_path_se':float(gap.std(ddof=1)/np.sqrt(len(gap))),
            'path_nll':model_path.tolist()}


def train_arm(seed,arm,args,initial,evals,outdir):
    model=V.VolPFN(**V.SIZES['tiny']); model.load_state_dict(initial); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    target_rng=np.random.default_rng(700000+seed)
    step=0; train_seconds=0.; diagnostic_seconds=0.; curve=[]
    start=time.perf_counter()
    while train_seconds < args.seconds:
        t=time.perf_counter()
        r,h,_=simulate(seed*100000+step,args.batch,args.length)
        f,b=batch_tensors(r)
        out=model(f,b); o={k:v[:,:-1] for k,v in out.items()}
        if arm=='standard':
            loss=-V.mixture_logpdf(o,torch.tensor(r[:,1:],dtype=torch.float32)).mean()
        else:
            x=np.sqrt(h[:,1:,None])*target_rng.standard_normal((args.batch,args.length-1,args.draws))
            loss=conditional_loss(o,torch.tensor(x,dtype=torch.float32))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        train_seconds += time.perf_counter()-t; step+=1
        if step%args.eval_every==0 or train_seconds>=args.seconds:
            d=time.perf_counter()
            row={'step':step,'training_seconds':train_seconds,'loss':float(loss.detach()),'id':evaluate(model,evals['id'],args.burn),'ood':evaluate(model,evals['ood'],args.burn)}
            diagnostic_seconds+=time.perf_counter()-d; curve.append(row)
    result={'seed':seed,'arm':arm,'steps':step,'training_seconds':train_seconds,'diagnostic_seconds':diagnostic_seconds,'wall_seconds':time.perf_counter()-start,'curve':curve}
    torch.save({'state_dict':model.state_dict(),'cfg':model.cfg,'config':vars(args),'seed':seed,'arm':arm},outdir/f'{seed}_{arm}.pt')
    (outdir/f'{seed}_{arm}.json').write_text(json.dumps(result,indent=2))
    print(f"seed={seed} {arm}: {step} steps, {train_seconds:.2f}s training incl simulation/targets; ID gap {curve[-1]['id']['gap_to_grid_reference']:.5f}; OOD NLL {curve[-1]['ood']['nll']:.5f}",flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seconds',type=float,default=20); ap.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]); ap.add_argument('--draws',type=int,default=8)
    ap.add_argument('--batch',type=int,default=8); ap.add_argument('--length',type=int,default=96); ap.add_argument('--burn',type=int,default=32); ap.add_argument('--out',default='research/outputs/pilot_v1')
    ap.add_argument('--eval-size',type=int,default=64); ap.add_argument('--eval-seed',type=int,default=990001); ap.add_argument('--eval-every',type=int,default=25)
    args=ap.parse_args()
    if args.eval_size<2 or args.eval_every<1 or args.seconds<=0 or args.draws<1 or args.batch<1 or not 1<=args.burn<args.length or len(set(args.seeds))!=len(args.seeds): ap.error('Invalid budget, shapes, burn, or duplicate seeds')
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata={'config':vars(args),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'platform':platform.platform(),
              'grid':parameter_grid().tolist(),'heldout_grid':parameter_grid(True).tolist(),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/grid_reference.py'),Path('volpfn.py'),Path('volpfn_prior.py')]},
              'protocol':'Exploratory. CPU single thread. Train-time includes simulation, features, targets, backward and optimizer. Common diagnostic evaluation excluded and separately reported. One-step budget overshoot allowed. OOD reference is deliberately misspecified; not a Bayes-optimal OOD oracle.'}
    (outdir/'protocol.json').write_text(json.dumps(metadata,indent=2))
    source=outdir/'source'; source.mkdir()
    import shutil
    for name in ('research/pilot.py','research/grid_reference.py','volpfn.py','volpfn_prior.py'): shutil.copy2(name,source/Path(name).name)
    evals={'id':make_eval(args.eval_seed,args.eval_size,args.length),'ood':make_eval(args.eval_seed+1,args.eval_size,args.length,True)}
    results=[]
    for i,seed in enumerate(args.seeds):
        torch.manual_seed(seed); initial=copy.deepcopy(V.VolPFN(**V.SIZES['tiny']).state_dict())
        arms=['standard','conditional'] if i%2==0 else ['conditional','standard']
        for arm in arms: results.append(train_arm(seed,arm,args,initial,evals,outdir))
    differences={}
    for metric in ('id','ood'):
        ds=[]
        for seed in args.seeds:
            pair={r['arm']:r['curve'][-1][metric]['nll'] for r in results if r['seed']==seed}
            ds.append(pair['conditional']-pair['standard'])
        differences[metric]={'conditional_minus_standard_by_seed':ds,'mean':float(np.mean(ds)),'seed_se':float(np.std(ds,ddof=1)/np.sqrt(len(ds))) if len(ds)>1 else None}
    (outdir/'summary.json').write_text(json.dumps(differences,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for row in results:
        color='tab:blue' if row['arm']=='standard' else 'tab:orange'
        for ax,metric,label in zip(axes,('id','ood'),('In-prior predictive NLL','Withheld-parameter predictive NLL')):
            ax.plot([x['training_seconds'] for x in row['curve']],[x[metric]['nll'] for x in row['curve']],color=color,alpha=.65,label=row['arm'] if row['seed']==args.seeds[0] else None)
            ax.set_xlabel('Training seconds, including simulation and targets'); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Exploratory tiny-model pilot — lower is better'); fig.tight_layout(); fig.savefig(outdir/'accuracy_compute.png',dpi=160)
    print(json.dumps(differences,indent=2))
if __name__=='__main__': main()

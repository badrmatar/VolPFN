"""Display the frozen score summary; does not fit models or read raw market data."""
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.market_density.run import OUTPUT


def main():
    results=json.loads((OUTPUT/'summary.json').read_text())
    labels=['Bayesian log-HAR','Bootstrap log-HAR','Gaussian log-HAR-GARCH',
            'Bootstrap log-HAR-GARCH','Gaussian root-HAR-GARCH']
    colors=['#3366aa','#88929c','#cf8432','#24887d','#985da0']
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharey=True)
    for row,proxy in enumerate(['rv5','rk_parzen']):
        result=results[proxy]['groups']['all'];p=np.array(result['point'])[:,2,:];ci=np.array(result['ci95'])[:,:,2,:]
        for col,(metric,mult,title,reference) in enumerate([(1,100,'Central 95% interval coverage (%)',95),(3,100,'Upper 1% violations (%)',1),(0,1,'Normalized quantile loss (lower is better)',None)]):
            ax=axes[row,col]
            for m in range(5):
                ax.plot([mult*ci[0,m,metric],mult*ci[1,m,metric]],[m,m],color=colors[m],lw=2)
                ax.scatter(mult*p[m,metric],m,color=colors[m],s=42,zorder=3)
            if reference is not None:ax.axvline(reference,color='#333333',ls='--',lw=1)
            ax.set_yticks(range(5));ax.set_yticklabels(labels);ax.set_ylim(4.7,-.7)
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            for side in ['top','right','left']:ax.spines[side].set_visible(False)
            ax.tick_params(axis='y',length=0)
            if row==0:ax.set_title(title,fontsize=11,pad=15)
            if col==0:ax.set_ylabel('RV5' if row==0 else 'Parzen RK',fontsize=12,labelpad=12)
    fig.suptitle('Real-market 22-observation variance forecasts',fontsize=17,x=.55,y=.99)
    fig.text(.55,.945,'28 markets · 2011–2014 development period · five established model controls',ha='center',fontsize=11,color='#555555')
    fig.text(.03,.025,'Bars: exploratory 95% intervals from joint calendar-date blocks. Dashed lines: nominal rates.\nModels are fitted separately to each measure; these are variance quantiles, not return VaR.',fontsize=10,color='#555555')
    fig.tight_layout(rect=(0,.085,1,.91),h_pad=3,w_pad=2)
    fig.savefig(OUTPUT/'density_diagnostic.png',dpi=180)
    fig.savefig(OUTPUT/'density_diagnostic.pdf')
    (OUTPUT/'visualization_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUTPUT/'summary.json']},indent=2))


if __name__=='__main__':main()

from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
h=json.loads((ROOT/'historical_analysis_v1/statistics.json').read_text())['contrasts']
t=json.loads((ROOT/'ttm_control_v1/statistics.json').read_text())['contrasts']
e=json.loads((ROOT/'external_confirmation_v1/statistics.json').read_text())['contrasts']
fig,axes=plt.subplots(3,1,figsize=(10,8.8),gridspec_kw={'height_ratios':[2,4,3]},layout='constrained')
fig.get_layout_engine().set(rect=(0,.055,1,.945))
sets=[([x for x in h if x['block']==60],lambda x:'Synthetic '+x['synthetic']+' vs historical','2 · Stronger historical comparison'),
      ([x for x in t if x['block']==60],lambda x:'Blend vs '+x['control'].replace('ttm_','TTM ')+' ('+('return' if x['proxy']=='r2' else 'kernel')+')','3 · TTM calibration controls'),
      ([x for x in e if x['block']==60],lambda x:x['reference']+' vs '+('rolling GJR' if x['comparator'].startswith('gjr') else 'historical'),'4 · New ETF population')]
for ax,(rows,label,title) in zip(axes,sets):
    for i,x in enumerate(rows):
        lo,hi=x['family_interval95'];gain=x['gain'];color='#087f5b' if lo>0 else '#b26c12'
        ax.errorbar(gain,i,xerr=[[gain-lo],[hi-gain]],fmt='o',color=color,capsize=4,markersize=6,linewidth=1.8)
    ax.axvline(0,color='#444444',linestyle='--',linewidth=.8)
    ax.set_yticks(range(len(rows)),[label(x) for x in rows]);ax.invert_yaxis()
    ax.set_title(title,loc='left',weight='bold');ax.grid(axis='x',alpha=.2);ax.set_xlim(-.012,.045)
axes[-1].set_xlabel('Comparator QLIKE minus reference QLIKE (positive favours the named reference)')
fig.suptitle('Primary estimates and family-adjusted confidence intervals',weight='bold',fontsize=14)
fig.text(.01,.015,'20,000 synchronized calendar resamples · 60-date blocks · uncertainty conditional on saved forecasts',fontsize=9)
fig.savefig(ROOT/'primary_comparisons.png',dpi=200,bbox_inches='tight')

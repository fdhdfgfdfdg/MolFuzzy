"""Regenerate manuscript figures from validation/results.json; no invented values."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    d=json.loads((ROOT/'validation/results.json').read_text());r=d['case']
    out=ROOT/'docs/figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    ax[0].bar(r['criterion_labels'],r['criteria']['weights'],color='#375a7f')
    ax[0].set(ylim=(0,.30),ylabel='Normalized weight',title='A  Criterion similarity network')
    order=r['ranking']['ranking']; names=r['alternative_labels'];scores=r['ranking']['scores']
    ax[1].bar([names[i] for i in order],[scores[i] for i in order],color='#287d71')
    ax[1].set(ylim=(0,1),ylabel='Additive utility',title='B  Preference scores')
    for a in ax:
        for rect in a.patches:
            a.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.009,f'{rect.get_height():.4f}',ha='center',fontsize=9)
    fig.savefig(out/'fig1_case.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,3.4),layout='constrained')
    initial=r['balancing']['residual_history'][0]
    for lr in (.1,.5,1.):
        record=next(x for x in d['sensitivity'] if x['learning_rate']==lr and x['geometry']=='linear')
        n=record['balance_iterations'];t=np.arange(n+1)
        residual=initial*(1-lr)**t
        ax.semilogy(t,np.maximum(residual,1e-9),label=f'lambda={lr:g}, {n} iterations',linewidth=1.8)
    ax.axhline(1e-6,color='black',linestyle='--',linewidth=1,label='Residual tolerance')
    ax.set(xlabel='Relaxation iteration',ylabel='Maximum target residual',ylim=(1e-9,1),title='Consensus speed changes; analytic target is unchanged')
    ax.legend(fontsize=9)
    fig.savefig(out/'fig2_consensus.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,3.3),layout='constrained')
    exact=r['sequence']['path_cost']; trials=d['heuristic_100_seeds']
    gaps=[100*(x['path_cost']/exact-1) for x in trials]
    ax.scatter([x['seed'] for x in trials],gaps,s=19,color='#375a7f')
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(xlabel='Random seed',ylabel='Cost gap above exact optimum (%)',title='Optional heuristic traversal: 100 independent seeds')
    fig.savefig(out/'fig3_traversal.png',dpi=220);plt.close(fig)
    print('Saved 3 figures from measured records.')

if __name__=='__main__':main()

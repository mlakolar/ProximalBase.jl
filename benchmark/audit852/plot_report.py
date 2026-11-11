#!/usr/bin/env python3
"""Standalone report figures from exported, verified summaries only."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':9,'svg.hashsalt':'852-final'})
inputs={}
def rows(context):
    f=a.results/context/'summary.csv';inputs[str(f.relative_to(a.results))]=hashlib.sha256(f.read_bytes()).hexdigest()
    return {r['case']:r for r in csv.DictReader(f.open()) if r['before']=='original' and r['after']=='post850'}
def save(fig,name):
    fig.savefig(a.output/(name+'.svg'),metadata={'Date':None});fig.savefig(a.output/(name+'.png'),dpi=170);plt.close(fig)
def curve(ax,points,label,color,marker='o',ls='-'):
    if not points:return
    points.sort();x=[v[0] for v in points];y=[float(v[1]['ratio']) for v in points]
    low=[v-float(r['ci95_low']) for v,(_,r) in zip(y,points)]
    high=[float(r['ci95_high'])-v for v,(_,r) in zip(y,points)]
    ax.errorbar(x,y,yerr=[low,high],label=label,color=color,marker=marker,ls=ls,capsize=2)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for context,label,color in [('broad7','Native copy setup, 7 rounds','#7570b3'),('common-full7','Common full setup, 7 rounds','#d95f02'),('common-active7','Common active setup, 7 rounds','#1b9e77')]:
    data=rows(context)
    for ax,op in zip(axes,['cd_fixed','dot']):
        pts=[(s,data[f'sparse/Float64/n1000000/s{s}/b1/scalar/{op}']) for s in [100,1000,10000] if f'sparse/Float64/n1000000/s{s}/b1/scalar/{op}' in data]
        curve(ax,pts,label,color)
for ax,op in zip(axes,['Four diagonal CD sweeps','Dense-vector dot']):
    ax.axhline(1,color='black',lw=.8);ax.set(xscale='log',xlabel='Active coordinates (n = 1,000,000)',ylabel='Post850 / original',title=op);ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Scalar preparation changes observed ratios; contexts are not pooled')
save(fig,'scalar-preparation')

data=rows('supplement-common-full-confirmation21')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,pct in zip(axes,[1,10]):
    for layout,color in [('contiguous','#2166ac'),('strided','#b2182b')]:
        pts=[]
        for b in [1,8,64]:
            s=max(1,(8192//b)*pct//100);key=f'block/Float64/n8192/s{s}/b{b}/{layout}/residual_bcd'
            if key in data:pts.append((b,data[key]))
        curve(ax,pts,layout,color)
    ax.axhline(1,color='black',lw=.8);ax.set(xscale='log',xlabel='Block size',ylabel='Post850 / original',title=f'{pct}% target active blocks');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Four residual-maintaining BCD sweeps: native access, confirmation21')
save(fig,'residual-bcd')

fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for psize,color in [(144,'#2166ac'),(1414,'#b2182b')]:
    pts=[]
    for s in [100,1000,10000]:
        key=f'symmetric/Float64/n{psize}/s{s}/b1/triangle/bilinear_native'
        if key in data:pts.append((s,data[key]))
    curve(axes[0],pts,f'p={psize}',color)
axes[0].axhline(1,color='black',lw=.8);axes[0].set(xscale='log',xlabel='Active independent coordinates',ylabel='Post850 / original',title='Optimized historical bilinear helper');axes[0].legend();axes[0].grid(alpha=.2)
full=rows('common-full7')
for label,color,field in [('original','#8c510a','before_ns'),('post850','#01665e','after_ns')]:
    for n,ls in [(10000,'--'),(1000000,'-')]:
        pts=[]
        for s in [100,1000,10000]:
            key=f'sparse/Float64/n{n}/s{s}/b1/scalar/clear'
            if key in full:pts.append((s,float(full[key][field])/1000))
        if pts:axes[1].plot(*zip(*pts),marker='o',color=color,ls=ls,label=f'{label}, n={n}')
axes[1].set(xscale='log',yscale='log',xlabel='Active coordinates',ylabel='Microseconds / clear',title='Sparse clear, common full setup (7 rounds)');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
fig.suptitle('Active-set proportional kernels: source/helper context is explicit')
save(fig,'active-kernels')
(a.output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
print('Wrote 3 standalone SVG/PNG figures and input summary hashes.')

#!/usr/bin/env python3
"""Map every exploratory >5% flag to its independent confirmation and context."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
def read(name):return list(csv.DictReader((a.results/name/'summary.csv').open()))
contexts=[('broad7','native-confirmation21',False),('common-full7','common-full-confirmation21',False),
 ('common-active7','common-active-confirmation21',False),
 ('supplement-common-full-exploratory7','supplement-common-full-confirmation21',False),
 ('raw-block7','raw-block-confirmation21',False),
 ('supplement-native7','supplement-common-full-confirmation21',True)]
if (a.results/'static-adapter7').exists():contexts.append(('static-adapter7','static-adapter-confirmation21',False))
# Supplemental28 selection was registered from native7 flags plus all residual
# and native-helper cases; newly exposed flags within that selection also have
# the independently analyzed last21.
ledger=[]
for first,follow,changed_context in contexts:
    rows=read(first);flags=[r for r in rows if r['after']=='post850' and r['before'] in ('original','post848') and float(r['ratio'])>1.05]
    followpath=a.results/follow/'summary.csv'
    assert followpath.exists() or not flags,(follow,'confirmation missing')
    confirmations={(r['case'],r['before'],r['after']):r for r in (read(follow) if flags else [])}
    for r in flags:
        key=(r['case'],r['before'],r['after']);assert key in confirmations,('unconfirmed flag',first,key)
        c=confirmations[key];assert int(c['rounds'])>=20
        fam,precision,n,s,b,layout,op=r['case'].split('/')
        if r['before']=='post848':classification='identical executable iterate AST; compiler/fixture/process context investigated with A/A'
        elif min(float(c['before_ns']),float(c['after_ns']))<100:classification='timer-limited; interpret with batched/sequence evidence'
        elif op.startswith('construct') or op=='dense_import':
            if fam=='block':classification='block layout/map/activity construction; inspect increased bytes/counts and repeated timings'
            elif any(float(c['before_'+field])!=float(c['after_'+field]) for field in ('bytes','allocs')):
                classification='constructor/import allocation changes; inspect bytes/counts and process controls'
            else:classification='matching scalar allocations; inspect isolated A/A and allocation/process context'
        elif fam=='block' and op in ('copyto','copyto_shared'):classification='layout equality and per-active-group state replacement'
        elif fam=='block' and op in ('block_cd','block_sweep','write_active','repeat_update','residual_bcd'):classification='owner-view writes and group activity tracking; inspect native-index controls'
        elif fam=='symmetric' and op in ('bilinear','matmat','matvec'):classification='product dispatch and stored-coordinate kernel; inspect native helper and common setup'
        else:classification='scalar/cache context; inspect common-full and common-active controls and A/A'
        repeated=float(c['ratio'])>1.05
        if repeated and float(c['ci95_low'])>1.05:outcome='confirmation interval above 1.05 in this context'
        elif repeated:outcome='confirmation median above 1.05; interval does not establish >5%'
        else:outcome='initial >5% median not repeated in confirmation median'
        ledger.append(dict(exploratory_context=first,confirmation_context=follow,
          context_change=('common restoration/longer budget; native indexing also changes residual BCD' if changed_context else 'same measured operation and restoration protocol'),
          case=r['case'],before=r['before'],after=r['after'],
          exploratory_ratio=r['ratio'],confirmation_rounds=c['rounds'],confirmation_ratio=c['ratio'],
          confirmation_ci95_low=c['ci95_low'],confirmation_ci95_high=c['ci95_high'],
          before_ns=c['before_ns'],after_ns=c['after_ns'],before_bytes=c['before_bytes'],after_bytes=c['after_bytes'],
          before_allocs=c['before_allocs'],after_allocs=c['after_allocs'],aa_ratio=c['aa_ratio'],
          pairs_above_5=c['pairs_above_5'],outcome=outcome,investigation=classification))
with (a.results/'regression-ledger.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
print(len(ledger),'flagged contrasts have >=20 follow-up rounds; context changes are explicit')

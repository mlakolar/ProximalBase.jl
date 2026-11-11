#!/usr/bin/env python3
"""Copy verified summaries and emit a single campaign-indexed results table."""
import argparse,csv,hashlib,json,os,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--broad',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();prefix=str(a.prefix.resolve());out=a.output;out.mkdir(parents=True,exist_ok=True)
adapter=json.loads(Path(prefix+'-adapter-control-complete.json').read_text())
assert adapter['status']=='pass'
paths=[('broad7',a.broad,Path(prefix+'-broad-summary')),
 ('native-confirmation21',Path(prefix+'-followup'),Path(prefix+'-followup-summary')),
 ('supplement-native7',Path(prefix+'-supplement'),Path(prefix+'-supplement-summary')),
 ('common-full7',Path(prefix+'-restore'),Path(prefix+'-restore-summary')),
 ('supplement-common-full28',Path(prefix+'-supplement-restore'),Path(prefix+'-supplement-restore-summary')),
 ('common-active7',Path(prefix+'-active-restore'),Path(prefix+'-active-restore-summary')),
 ('raw-block7',Path(prefix+'-raw-block'),Path(prefix+'-raw-block-summary')),
 ('raw-block-confirmation21',Path(prefix+'-raw-block-confirmation'),Path(prefix+'-raw-block-confirmation-summary')),
 ('backing-blas7',Path(prefix+'-backing-blas'),Path(prefix+'-backing-blas-summary')),
 ('backing-blas-confirmation21',Path(prefix+'-backing-blas-confirmation'),Path(prefix+'-backing-blas-confirmation-summary')),
 ('common-full-confirmation21',Path(prefix+'-restore-confirmation'),Path(prefix+'-restore-confirmation-summary')),
 ('common-active-confirmation21',Path(prefix+'-active-restore-confirmation'),Path(prefix+'-active-restore-confirmation-summary'))]
if adapter['static_control_measured']:
    paths.append(('static-adapter7',Path(prefix+'-static-adapter'),Path(prefix+'-static-adapter-summary')))
    if Path(prefix+'-static-adapter-summary/suspicious.txt').read_text().strip():
        paths.append(('static-adapter-confirmation21',Path(prefix+'-static-adapter-confirmation'),Path(prefix+'-static-adapter-confirmation-summary')))
for i in range(3):
 base=Path(prefix+'-allocation-isolation');paths.append((f'allocation-isolation-{i:02}',base/f'case-{i:02}',base/f'summary-{i:02}'))
allrows=[];index=[]
for name,raw,summary in paths:
    if not raw.exists():
        # A confirmation may be empty only when its exploratory flag file is empty.
        context='raw-block' if name=='raw-block-confirmation21' else 'restore' if name=='common-full-confirmation21' else 'active-restore'
        if name=='raw-block-confirmation21' and not Path(prefix+'-raw-block-summary/suspicious.txt').read_text().strip():continue
        if name=='backing-blas-confirmation21' and not Path(prefix+'-backing-blas-summary/suspicious.txt').read_text().strip():continue
        if name.endswith('confirmation21') and name.startswith('common-'):
            assert not Path(prefix+'-'+context+'-summary/suspicious.txt').read_text().strip()
            continue
        raise FileNotFoundError(raw)
    analysis=json.loads((summary/'analysis.json').read_text());assert analysis['raw_samples_verified']
    assert analysis['observations_sha256']==hashlib.sha256((raw/'observations.csv').read_bytes()).hexdigest()
    dest=out/name;dest.mkdir(exist_ok=True)
    for f in summary.iterdir():
        if f.is_file():shutil.copyfile(f,dest/f.name)
    rows=list(csv.DictReader((summary/'summary.csv').open()))
    annotated=[]
    for r in rows:
        affected=(adapter['static_control_measured'] and not name.startswith('static-adapter')
          and 'historical05' in (r['before'],r['after']) and r['case'].startswith('block/'))
        status='superseded for source attribution: static API selection and equal retained-zero copy preparation' if affected else 'comparable source observation in the stated context'
        annotated.append(dict(source_attribution=status,**r))
        allrows.append(dict(campaign=name,source_attribution=status,**r))
    with (dest/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(annotated[0]));writer.writeheader();writer.writerows(annotated)
    compatibility=list(csv.DictReader((summary/'incompatibilities.csv').open()))
    for row in compatibility:
        error=row['error'];op=row['case'].rsplit('/',1)[1]
        if 'original block public constructor' in error:
            scope=('Construction scope: native row factories are separate; raw-field factory timing is intentionally excluded'
              if op.startswith('construct') else 'Convenience-vector API exclusion only; see raw-native and native-row campaigns for representable original operations')
        elif 'no active group metadata' in error:scope='Genuine original feature incompatibility: no active-group metadata'
        elif 'historical fill zero removes' in error:scope='Different activity semantics; explicit zero-write/reconstruction sequences remain comparable'
        elif 'full active set' in error:scope='Fixture has no inactive coordinates; no measurement is inferred'
        else:scope='Explicit recorded capability/scope exclusion; see the original reason'
        row['audit_scope']=scope
    if compatibility:
        with (dest/'incompatibilities.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(compatibility[0]));writer.writeheader();writer.writerows(compatibility)
    config=json.loads((raw/'campaign.json').read_text());counts={}
    for r in csv.DictReader((raw/'observations.csv').open()):counts[r['status']]=counts.get(r['status'],0)+1
    assert counts.get('failure',0)==0
    if name in ('broad7','native-confirmation21'):harness_relative='MAIN'
    else:
        if name.startswith('allocation-isolation'):harness=raw.parent/'harness'
        elif name.startswith('raw-block'):harness=Path(prefix+'-raw-block-harness')
        elif name.startswith('static-adapter'):harness=Path(prefix+'-static-adapter-harness')
        elif name.startswith('backing-blas'):harness=Path(prefix+'-backing-blas-harness')
        else:harness=raw.with_name(raw.name+'-harness')
        assert harness.exists();harness_relative=os.path.relpath(harness,Path(prefix).parent)
    index.append(dict(campaign=name,raw_directory=raw.name,analysis_harness=harness_relative,
        raw_relative_directory=os.path.relpath(raw,Path(prefix).parent),summary_directory=summary.name,
        rounds=config['argv']['rounds'],start_round=config['argv']['start_round'],
        chosen_cases=config['argv']['chosen_case_count'],roles=config['argv']['roles'],
        pass_rows=counts.get('pass',0),incompatible_rows=counts.get('incompatible',0),
        observations_sha256=analysis['observations_sha256']))
for name,rows in [('campaigns.csv',index),('all-comparisons.csv',allrows)]:
    with (out/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
excluded=[r for r in allrows if r['source_attribution'].startswith('superseded')]
with (out/'adapter-exclusions.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(excluded)
shutil.copyfile(prefix+'-adapter-control-complete.json',out/'adapter-diagnostic.json')
# Partitions are derived from the same supplemental28, not extra measurements.
partitions=Path(prefix+'-supplement-partitions')
for name in ('exploratory7','confirmation21'):
    src=partitions/name;assert json.loads((src/'analysis.json').read_text())['raw_samples_verified']
    shutil.copytree(src,out/('supplement-common-full-'+name),dirs_exist_ok=True)
(out/'README.md').write_text('''# Machine-readable results

campaigns.csv identifies raw campaign names, round ranges, role sets, expected
case counts and observations SHA256. all-comparisons.csv concatenates every
verified comparison with an explicit campaign column. Per-campaign summary.csv
and tables.md include all comparable pairs; incompatibilities.csv gives each
excluded case/role and its reason. Nothing is inferred for missing capabilities.

R is the median paired round ratio; marginal before/after times need not divide
to R. A/A comparisons appear as their own rows, with confidence intervals.
Allocation counts/bytes are trial estimates. Reachable memory is summarysize.
Native, common-full, common-active and native-index contexts are separate.
The source_attribution column marks any historical05 view-operation contrasts
invalidated by the API-selection diagnostic. Original raw rows are retained;
adapter-exclusions.csv lists affected contrasts and static-adapter7 replaces
their source attribution when the diagnostic confirms an artifact.
The supplemental exploratory7 and confirmation21 directories partition the
same full28 campaign; they are not additional raw observations.

Broad SVG/PNG scaling figures use native copy setup. See report.md for the
matched-preparation figures and limitations before interpreting scalar gains.
''')
print(len(index),'campaigns;',len(allrows),'comparison rows exported')

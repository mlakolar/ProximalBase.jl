#!/usr/bin/env python3
"""Report exploratory7 and confirmation21 from a verified 28-round campaign."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('harness',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();sys.path.insert(0,str(a.harness.resolve()))
spec=importlib.util.spec_from_file_location('campaign_analysis',a.harness/'analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=module.load(a.campaign);module.verify_completeness(a.campaign,rows);module.verify_samples(a.campaign,rows)
rounds=sorted({int(r['round']) for r in rows});assert len(rounds)==28,rounds
for name,chosen in [('exploratory7',rounds[:7]),('confirmation21',rounds[7:])]:
    output=a.output/name;output.mkdir(parents=True,exist_ok=True)
    selected=[r for r in rows if int(r['round']) in chosen]
    summary=module.summarize(selected,10000)
    module.writecsv(output/'summary.csv',summary);module.tables(output,summary)
    (output/'analysis.json').write_text(json.dumps({'campaign':a.campaign.name,
      'observations_sha256':hashlib.sha256((a.campaign/'observations.csv').read_bytes()).hexdigest(),
      'rounds':chosen,'raw_samples_verified':True,'rows':len(selected),'comparisons':len(summary),
      'bootstrap_seed':852,'bootstrap_draws':10000,'unit':'paired process round'},indent=2)+'\n')
print('Verified 28-round campaign; wrote separate exploratory7 and confirmation21 summaries.')

"""Select balanced, successfully judged objects; never retain judge failures."""
import hashlib,json
from collections import Counter
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
summary={}
for split,n in [('train',1000),('validation',100)]:
 raw=ROOT/f'datasets/galaxy_json_pilot_v1_{split}_raw';scored=ROOT/f'datasets/galaxy_json_pilot_v1_{split}_scored'
 scores=pd.read_parquet(scored/'scores.parquet');data=pd.read_parquet(raw/'data.parquet');manifest=json.loads((raw/'manifest.json').read_text());samples=manifest['samples']
 good=scores[(scores['status']=='scored')&(scores['score']>=7)]
 pools={}
 for _,r in good.iterrows():
  i=int(r.sample_index);cat=samples[i]['category'];pools.setdefault(cat,[]).append(i)
 assert sum(map(len,pools.values()))>=n, f'Only {len(good)} qualified {split} samples; need {n}'
 selected=[]
 while len(selected)<n:
  for cat in sorted(pools):
   if pools[cat] and len(selected)<n:selected.append(pools[cat].pop(0))
 rows=data.iloc[selected].reset_index(drop=True);out=ROOT/f'datasets/galaxy_json_pilot_v1_{split}_filtered';out.mkdir(exist_ok=True)
 assert not (out/'data.parquet').exists(),'Immutable dataset already exists'
 rows.to_parquet(out/'data.parquet',index=False)
 chosen=[]
 for j,i in enumerate(selected):
  x=dict(samples[i],sample_index=j,source_sample_index=i);chosen.append(x)
 manifest['samples']=chosen;manifest['split']=split+'_filtered';manifest['filter']='judge:medium, score>=7, status=scored; balanced round-robin selection'
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 scores[scores.sample_index.isin(selected)].to_parquet(out/'scores.parquet',index=False)
 summary[split]={'count':n,'qualified_pool':len(good),'judge_failures_excluded':int((scores.status!='scored').sum()),'categories':dict(Counter(s['category'] for s in chosen)),'dataset_sha256':hashlib.sha256((out/'data.parquet').read_bytes()).hexdigest()}
sets=[];hashsets=[]
for split in ['train_filtered','validation_filtered','test']:
 s=json.loads((ROOT/f'datasets/galaxy_json_pilot_v1_{split}/manifest.json').read_text())['samples'];sets.append({x['object_id'] for x in s});hashsets.append({x['image_sha256'] for x in s})
for i in range(3):
 for j in range(i):assert not sets[i]&sets[j];assert not hashsets[i]&hashsets[j]
assert len(sets[2])==100
summary['test']={'count':100,'locked':True};summary['object_and_image_hash_disjoint']=True
(ROOT/'training/lqh/json-pilot-v1/final-data-audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

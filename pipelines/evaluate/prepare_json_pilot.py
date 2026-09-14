"""Deterministic catalog-derived JSON pilot; splits precede judge filtering."""
import base64,csv,hashlib,importlib.util,io,json,random
from collections import Counter
from pathlib import Path
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('retrieval',ROOT/'services/retrieval/match.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Reuse the exact baseline labeling function without executing its data selection.
source=(ROOT/'pipelines/evaluate/prepare_json_baseline.py').read_text();exec(source[source.index('def label(row):'):source.index('picked=[]')])
prompt=(ROOT/'prompts/galaxy_json_v1.md').read_text();question='Classify the central object in this image. Return only the morphology JSON specified in the system instructions.'
smoke=json.loads((ROOT/'datasets/galaxy_json_smoke_v1/manifest.json').read_text())['samples'];smoke_ids={x['object_id'] for x in smoke}
old_split=(ROOT/'data/splits/galaxy-zoo-2-pilot-v1.json').read_text()
image_dir=ROOT/'data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images';mapping={}
with (ROOT/'data/raw/galaxy-zoo-2/gz2_filename_mapping.csv').open() as f:
 for r in csv.DictReader(f):
  if r['objid'] not in mapping and (image_dir/(r['asset_id']+'.jpg')).is_file():mapping[r['objid']]=r['asset_id']
def category(q):
 v=q['features']
 if v['appearance'] is None:return 'ambiguous'
 if v['appearance']=='smooth':return 'smooth_round' if v['roundness']=='completely_round' else 'smooth_elongated'
 if v['edge_on']=='yes':return 'edge_on'
 if v['spiral']=='yes':return 'barred_spiral' if v['bar']=='yes' else 'other_spiral'
 return 'disk_other'
cats=['smooth_round','smooth_elongated','edge_on','barred_spiral','other_spiral','disk_other','ambiguous']
pools={c:[] for c in cats};test_fixed=[]
with (ROOT/'data/raw/galaxy-zoo-2/gz2_hart16.csv').open() as f:
 for r in csv.DictReader(f):
  oid=r['dr7objid']
  if oid not in mapping or (oid in old_split and oid not in smoke_ids):continue
  q,counts=label(r);c=category(q)
  item={'object_id':oid,'asset_id':mapping[oid],'category':c,'reference':q,'raw_counts':counts}
  if oid in smoke_ids:test_fixed.append(item)
  else:pools[c].append(item)
rng=random.Random(20260910)
for p in pools.values():rng.shuffle(p)
seen_hashes=set();out={};bad=0
def materialize(item):
 global bad
 b=(image_dir/(item['asset_id']+'.jpg')).read_bytes();sha=hashlib.sha256(b).hexdigest()
 if sha in seen_hashes:return None
 try:
  im=Image.open(io.BytesIO(b));im.verify()
 except Exception:bad+=1;return None
 seen_hashes.add(sha);item=dict(item,image_sha256=sha)
 messages=[{'role':'system','content':prompt},{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(b).decode()}},{'type':'text','text':question}]},{'role':'assistant','content':json.dumps(item['reference'])}]
 return item,{'messages':json.dumps(messages),'audio':None,'tools':None}
# Lock test first, including all six previous probes. Remaining pools are disjoint.
for split,n in [('test',100),('validation_raw',150),('train_raw',1500)]:
 entries=[];rows=[]
 if split=='test':
  for item in sorted(test_fixed,key=lambda x:x['asset_id']):
   pair=materialize(item);assert pair;entries.append(pair[0]);rows.append(pair[1])
 i=0
 while len(entries)<n:
  cat=cats[i%len(cats)];i+=1;assert pools[cat],cat
  pair=materialize(pools[cat].pop())
  if pair:entries.append(pair[0]);rows.append(pair[1])
 for j,item in enumerate(entries):item['sample_index']=j
 d=ROOT/f'datasets/galaxy_json_pilot_v1_{split}';d.mkdir(exist_ok=True)
 assert not (d/'data.parquet').exists(),'Do not overwrite fixed experiment datasets'
 pd.DataFrame(rows).to_parquet(d/'data.parquet',index=False)
 manifest={'split':split,'seed':20260910,'reference_rule':'Raw conditional votes >=70% and >=10 answers, with parent branches enforced; null denotes insufficient, ambiguous or inapplicable evidence.','system_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'samples':entries}
 (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');out[split]={'rows':len(rows),'categories':dict(Counter(x['category'] for x in entries))}
 if split=='train_raw':
  d=ROOT/'datasets/galaxy_json_pilot_v1_filter_demo';d.mkdir(exist_ok=True);pd.DataFrame(rows[:14]).to_parquet(d/'data.parquet',index=False)
assert len(seen_hashes)==1750
(ROOT/'training/lqh/json-pilot-v1/preparation.json').write_text(json.dumps({'splits':out,'bad_images':bad,'note':'1500/150 candidate pools to allow filtering down to 1000/100; test is fixed at100 and never filtered by model success.'},indent=2))
print(json.dumps(out,indent=2))

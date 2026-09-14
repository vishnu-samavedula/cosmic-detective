"""Prepare reproducible zero-shot fixtures. No catalog answers enter user/system input."""
import base64,csv,importlib.util,json,random
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('retrieval',ROOT/'services/retrieval/match.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
out=ROOT/'datasets/galaxy_json_smoke_v1';out.mkdir(exist_ok=True)
image_dir=ROOT/'data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images'
mapping={}
with (ROOT/'data/raw/galaxy-zoo-2/gz2_filename_mapping.csv').open() as f:
 for r in csv.DictReader(f):
  if r['objid'] not in mapping and (image_dir/(r['asset_id']+'.jpg')).is_file():mapping[r['objid']]=r['asset_id']
with (ROOT/'data/raw/galaxy-zoo-2/gz2_hart16.csv').open() as f: rows=list(csv.DictReader(f))
random.Random(42).shuffle(rows)
split=(ROOT/'data/splits/galaxy-zoo-2-pilot-v1.json').read_text()
def label(row):
 vals={};counts={}
 for name,(prefix,answers) in m.GROUPS.items():
  c={a:int(row[f'{prefix}_{b}_count']) for a,b in answers.items()};n=sum(c.values());best=max(c,key=c.get)
  vals[name]=best if n>=10 and c[best]/n>=.7 else None
  counts[name]=c
 # Follow the actual questionnaire: downstream labels require a supported parent.
 if vals['appearance']!='smooth':vals['roundness']=None
 if vals['appearance']!='features_or_disk':
  for k in ['edge_on','bar','spiral','bulge_prominence','arms_winding','arms_number']:vals[k]=None
 if vals['edge_on']!='no':
  for k in ['bar','spiral','bulge_prominence','arms_winding','arms_number']:vals[k]=None
 if vals['spiral']!='yes':
  for k in ['arms_winding','arms_number']:vals[k]=None
 return {'schema_version':1,'features':vals},counts
picked=[];seen=set()
for row in rows:
 oid=row['dr7objid']
 if oid not in mapping or oid in split:continue
 target,counts=label(row);v=target['features']
 category='ambiguous_295305' if oid=='588015510636265731' else ('smooth_round' if v['appearance']=='smooth' and v['roundness']=='completely_round' else 'smooth_elongated' if v['appearance']=='smooth' and v['roundness'] in ['in_between','cigar_shaped'] else 'edge_on' if v['edge_on']=='yes' else 'barred_spiral' if v['spiral']=='yes' and v['bar']=='yes' else 'unbarred_spiral' if v['spiral']=='yes' and v['bar']=='no' else None)
 if category and category not in seen:
  seen.add(category);picked.append((row,target,counts,category))
 if len(seen)==6:break
assert len(picked)==6
question='Classify the central object in this image. Return only the morphology JSON specified in the system instructions.'
records=[];manifest=[]
for i,(r,target,counts,category) in enumerate(picked):
 asset=mapping[r['dr7objid']];uri='data:image/jpeg;base64,'+base64.b64encode((image_dir/(asset+'.jpg')).read_bytes()).decode()
 messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':uri}},{'type':'text','text':question}]},{'role':'assistant','content':json.dumps(target)}]
 records.append({'messages':json.dumps(messages),'audio':None,'tools':None})
 manifest.append({'sample_index':i,'object_id':r['dr7objid'],'asset_id':asset,'category':category,'reference':target,'raw_counts':counts})
pd.DataFrame(records).to_parquet(out/'data.parquet',index=False)
(out/'manifest.json').write_text(json.dumps({'seed':42,'reference_rule':'raw conditional votes: >=10 answers and >=70% agreement; enforce branch parents; null otherwise. These are noisy proxy labels, not astronomical truth.','samples':manifest},indent=2))
schema={'type':'object','additionalProperties':False,'required':['schema_version','features'],'properties':{'schema_version':{'type':'integer','const':1},'features':{'type':'object','additionalProperties':False,'required':list(m.GROUPS),'properties':{k:{'enum':list(v[1])+[None]} for k,v in m.GROUPS.items()}}}}
(ROOT/'prompts/galaxy_json_v1.output-schema.json').write_text(json.dumps(schema,indent=2))
prompt='''You classify the visible morphology of the central object in a galaxy image. Return ONLY a JSON object matching the schema below. No markdown, explanations, names, coordinates, or extra fields. No worked examples are provided.
Use null when a feature is ambiguous, not visible, or inapplicable. Do not guess hidden structures. appearance describes smooth versus features/disk versus star/artifact. roundness applies only to smooth objects. edge_on applies only to features_or_disk. bar, spiral and bulge_prominence apply only when appearance is features_or_disk and edge_on is no. arms_winding and arms_number apply only when spiral is yes. Every other downstream field must be null. A bright center does not necessarily mean a dominant bulge. cant_tell means arms are present but their number cannot be determined.
Schema:
'''+json.dumps(schema)
(ROOT/'prompts/galaxy_json_v1.md').write_text(prompt)
(ROOT/'evals/scorers/galaxy_json_v1.md').write_text('''Judge the assistant's morphology JSON against the image. Score 1-10 for visible feature plausibility, uncertainty and question-branch consistency. Invalid JSON is 1. Unsupported identity or physical claims are 1. Missing features should be null, not guessed. Do not demand certainty on ambiguous images. Return {"score": number, "reasoning": "brief evidence"}. Deterministic schema and reference comparisons are performed separately; this judge score is secondary.\n''')
print(json.dumps([{'image':x['asset_id']+'.jpg','category':x['category']} for x in manifest],indent=2))

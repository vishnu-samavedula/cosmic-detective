"""Deterministic schema, proxy-label and retrieval checks; no judge needed."""
import argparse,csv,importlib.util,json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('retrieval',ROOT/'services/retrieval/match.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

parser=argparse.ArgumentParser()
parser.add_argument('--run',default='baseline_json_v1')
parser.add_argument('--output-dir',default='artifacts/evaluations/baseline-json-v1')
parser.add_argument('--constrained',action='store_true')
parser.add_argument('--dataset',default='galaxy_json_smoke_v1')
args=parser.parse_args()
manifest=json.loads((ROOT/'datasets'/args.dataset/'manifest.json').read_text())
fields=list(m.GROUPS)
def check_schema(q):
 if not isinstance(q,dict) or set(q)!={'schema_version','features'} or type(q['schema_version']) is not int or q['schema_version']!=1:raise ValueError('wrong top-level schema')
 v=q['features']
 if not isinstance(v,dict) or set(v)!=set(fields):raise ValueError('missing/extra feature fields')
 for k,x in v.items():
  if x is not None and (not isinstance(x,str) or x not in m.GROUPS[k][1]):raise ValueError('invalid value: '+k)
 return v

def check(q):
 v=check_schema(q)
 invalid=[]
 if v['appearance']!='smooth':invalid+=['roundness']
 if v['appearance']!='features_or_disk':invalid+=['edge_on','bar','spiral','bulge_prominence','arms_winding','arms_number']
 if v['edge_on']!='no':invalid+=['bar','spiral','bulge_prominence','arms_winding','arms_number']
 if v['spiral']!='yes':invalid+=['arms_winding','arms_number']
 if any(v[k] is not None for k in invalid):raise ValueError('question-branch violation')
 return v
image_dir=ROOT/'data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images'
mapping={}
with (ROOT/'data/raw/galaxy-zoo-2/gz2_filename_mapping.csv').open() as f:
 for r in csv.DictReader(f):
  if r['objid'] not in mapping and (image_dir/(r['asset_id']+'.jpg')).is_file():mapping[r['objid']]=r['asset_id']
records=[];vectors=[];keys=[(k,a) for k,(_,answers) in m.GROUPS.items() for a in answers]
with (ROOT/'data/raw/galaxy-zoo-2/gz2_hart16.csv').open() as f:
 for r in csv.DictReader(f):
  oid=r['dr7objid']
  if oid not in mapping:continue
  records.append({'object_id':oid,'asset_id':mapping[oid],'gz2_class':r['gz2_class'],'ra':float(r['ra']),'dec':float(r['dec'])})
  vec=[]
  for k,(prefix,answers) in m.GROUPS.items():
   c=[int(r[f'{prefix}_{a}_count']) for a in answers.values()];n=sum(c)
   vec.extend([x/n if n>=10 else 0 for x in c])
  vectors.append(vec)
mat=np.asarray(vectors,dtype=np.float64);del vectors
ids=[r['object_id'] for r in records];indices={x:i for i,x in enumerate(ids)}
id_array=np.asarray(ids)
@lru_cache(maxsize=8)
def rank_query(cols):
 scores=mat[:,cols].mean(axis=1)
 order=np.lexsort((id_array,-scores))
 ranks=np.empty(len(order),dtype=np.int64);ranks[order]=np.arange(1,len(order)+1)
 return scores,order[:10],ranks
def retrieve(q,target):
 v=check(q);cols=[keys.index((k,x)) for k,x in v.items() if x is not None]
 if not cols:return {'status':'fallback','reason':'No supported features','description':'The image does not support a confident morphology description.'}
 scores,order,ranks=rank_query(tuple(cols))
 # Stable ID tie-break; score is morphology support, not identity probability.
 ti=indices[target];ts=scores[ti]
 return {'status':'possible_morphology_matches','warning':'No calibrated identity/no-match threshold; scores are conditional vote agreement.','top_score_ties':int(np.sum(np.isclose(scores,scores[order[0]],rtol=0,atol=1e-12))),'target_rank_with_id_tiebreak':int(ranks[ti]),'target_same_score':int(np.sum(np.isclose(scores,ts,rtol=0,atol=1e-12))),'target_in_top10':bool(ti in order),'candidates':[dict(records[i],morphology_agreement=float(scores[i])) for i in order]}
outputs={}
p=ROOT/'runs'/args.run/'results.parquet'
if p.exists():
 for _,r in pd.read_parquet(p).iterrows():
  msgs=r['messages'];msgs=json.loads(msgs) if isinstance(msgs,str) else msgs
  output=next(x['content'] for x in reversed(msgs) if x['role']=='assistant')
  outputs[int(r['sample_index'])]=output
results=[]
for sample in manifest['samples']:
 r={'image':sample['asset_id']+'.jpg','object_id':sample['object_id'],'category':sample['category'],'reference':sample['reference'],'reference_retrieval':retrieve(sample['reference'],sample['object_id'])}
 if sample['sample_index'] in outputs:
  raw=outputs[sample['sample_index']];r['model_raw_output']=raw
  try:
   q=json.loads(raw);check_schema(q);r['schema_valid']=True;v=check(q);r['valid_json_contract']=True;r['model_json']=q
   expected=sample['reference']['features'];known=[k for k,x in expected.items() if x is not None]
   r['reference_nonnull_fields']=len(known);r['matching_reference_nonnull_fields']=sum(v[k]==expected[k] for k in known)
   r['all_field_agreement']=sum(v[k]==expected[k] for k in fields)/len(fields)
   r['model_nonnull_fields']=sum(x is not None for x in v.values())
   r['model_retrieval']=retrieve(q,sample['object_id'])
  except (ValueError,TypeError,KeyError) as e:r['valid_json_contract']=False;r['error']=str(e);r['model_retrieval']={'status':'invalid_output','reason':'Do not retrieve from invalid JSON; retry or display inference error.'}
 results.append(r)
run_config=json.loads((ROOT/'runs'/args.run/'config.json').read_text())
report={'model':run_config.get('checkpoint_artifact_id') or run_config.get('hf_repo') or run_config.get('base_model'),'run':args.run,'mode':('constrained JSON decoding' if args.constrained else 'prompt-only JSON, no constrained decoding'),'catalog_size':len(records),'reference_policy':manifest['reference_rule'],'retrieval_policy':'Equal-weight mean raw vote agreement across predicted fields, minimum 10 votes per branch; absent evidence contributes zero; identity is not inferred by this score.','samples':results,'model_outputs_received':len(outputs)}
if outputs:
 valid=[r for r in results if r.get('valid_json_contract')]
 report['summary']={'schema_valid':sum(r.get('schema_valid',False) for r in results),'valid_contract':len(valid),'attempted':len(results),'matching_supported_reference_fields':sum(r['matching_reference_nonnull_fields'] for r in valid),'supported_reference_fields':sum(r['reference_nonnull_fields'] for r in valid),'source_object_in_model_top10':sum(r['model_retrieval'].get('target_in_top10',False) for r in valid), 'retrieval_attempted':sum(r['model_retrieval'].get('status')=='possible_morphology_matches' for r in valid), 'all_null_fallbacks':sum(r['model_retrieval'].get('status')=='fallback' for r in valid)}
report['reference_retrieval_summary']={'attempted':sum(r['reference_retrieval']['status']=='possible_morphology_matches' for r in results),'source_object_in_top10':sum(r['reference_retrieval'].get('target_in_top10',False) for r in results),'all_null_fallbacks':sum(r['reference_retrieval']['status']=='fallback' for r in results)}
out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
print(json.dumps([{'image':r['image'],'valid':r.get('valid_json_contract'),'reference_retrieval_status':r['reference_retrieval']['status'],'model_retrieval_status':r.get('model_retrieval',{}).get('status')} for r in results],indent=2))

"""JSON-to-Galaxy-Zoo retrieval smoke test. Standard library only.
Scores are morphology agreement, NOT probabilities of object identity.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    'arms_winding': ('t10_arms_winding', {'tight':'a28_tight','medium':'a29_medium','loose':'a30_loose'}),
    'arms_number': ('t11_arms_number', {'1':'a31_1','2':'a32_2','3':'a33_3','4':'a34_4','more_than_4':'a36_more_than_4','cant_tell':'a37_cant_tell'}),
    'appearance': ('t01_smooth_or_features', {'smooth':'a01_smooth','features_or_disk':'a02_features_or_disk','star_or_artifact':'a03_star_or_artifact'}),
    'roundness': ('t07_rounded', {'completely_round':'a16_completely_round','in_between':'a17_in_between','cigar_shaped':'a18_cigar_shaped'}),
    'edge_on': ('t02_edgeon', {'yes':'a04_yes','no':'a05_no'}),
    'bar': ('t03_bar', {'yes':'a06_bar','no':'a07_no_bar'}),
    'spiral': ('t04_spiral', {'yes':'a08_spiral','no':'a09_no_spiral'}),
    'bulge_prominence': ('t05_bulge_prominence', {'none':'a10_no_bulge','just_noticeable':'a11_just_noticeable','obvious':'a12_obvious','dominant':'a13_dominant'}),
}

def validate(query):
    if set(query) != {'schema_version','features'} or query['schema_version'] != 1:
        raise ValueError('Expected schema_version=1 and features only; IDs/names must not enter matching.')
    features = query['features']
    if not isinstance(features, dict) or not features or set(features) - GROUPS.keys():
        raise ValueError('Unknown or empty features.')
    for field, value in features.items():
        if value is not None and (not isinstance(value,str) or value not in GROUPS[field][1]):
            raise ValueError(f'Invalid {field}: use one of {list(GROUPS[field][1])} or null for uncertain/unobserved.')
    if not any(v is not None for v in features.values()):
        raise ValueError('No informative features; cannot rank catalog.')
    if features.get('appearance') == 'features_or_disk' and features.get('roundness') is not None:
        raise ValueError('GZ2 roundness is a smooth-branch question; leave it null for features/disk.')
    if features.get('edge_on') == 'yes' and any(features.get(k) is not None for k in ['bar','spiral','bulge_prominence']):
        raise ValueError('These GZ2 questions are on the not-edge-on branch; use null.')
    return {k:v for k,v in features.items() if v is not None}

def search(query, limit=10, target=None):
    selected=validate(query)
    if not 1 <= limit <= 10: raise ValueError('limit must be 1–10')
    mapping={}
    image_dir=ROOT/'data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images'
    with (ROOT/'data/raw/galaxy-zoo-2/gz2_filename_mapping.csv').open() as f:
        for r in csv.DictReader(f):
            if r['objid'] not in mapping and (image_dir/(r['asset_id']+'.jpg')).is_file():
                mapping[r['objid']]=r['asset_id']
    candidates=[]
    with (ROOT/'data/raw/galaxy-zoo-2/gz2_hart16.csv').open() as f:
        for r in csv.DictReader(f):
            oid=r['dr7objid']
            if oid not in mapping: continue
            evidence={}
            for field,value in selected.items():
                prefix,answers=GROUPS[field]
                counts={a:int(r[f'{prefix}_{suffix}_count']) for a,suffix in answers.items()}
                n=sum(counts.values())
                # Unanswered/sparsely answered branches are missing evidence, never negative labels.
                if n < 5: continue
                evidence[field]={'requested':value,'matching_votes':counts[value],'branch_votes':n,'agreement':counts[value]/n}
            coverage=len(evidence)/len(selected)
            score=sum(e['agreement'] for e in evidence.values())/len(selected)
            candidates.append({'object_id':oid,'asset_id':mapping[oid],'image':str(image_dir/(mapping[oid]+'.jpg')),'gz2_class':r['gz2_class'],'ra':float(r['ra']),'dec':float(r['dec']),'morphology_agreement':score,'evidence_coverage':coverage,'evidence':evidence})
    # ID resolves identical scores reproducibly, without claiming a meaningful order within ties.
    candidates.sort(key=lambda r:(-r['morphology_agreement'],-r['evidence_coverage'],r['object_id']))
    report={'query':query,'method':'Mean raw conditional vote agreement; missing branches contribute no evidence. Minimum five votes per branch. Equal feature weights.','warning':'Experimental morphology ranking, not identity confidence. No calibrated match/no-match threshold. IDs only join records and break ties.','searched_objects':len(candidates),'candidates':candidates[:limit]}
    if candidates:
        top=candidates[0]['morphology_agreement']
        report['objects_tied_at_top']=sum(abs(c['morphology_agreement']-top)<1e-12 for c in candidates)
    if target:
        for i,c in enumerate(candidates):
            if c['object_id']==target:
                s=c['morphology_agreement'];report['diagnostic_target']={'object_id':target,'rank_with_id_tiebreak':i+1,'strictly_better':sum(x['morphology_agreement']>s+1e-12 for x in candidates),'same_score':sum(abs(x['morphology_agreement']-s)<1e-12 for x in candidates),'record':c};break
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--output',required=True);p.add_argument('--target',help='Diagnostic only; never used to rank');a=p.parse_args()
    result=search(json.loads(Path(a.query).read_text()),target=a.target)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['candidates','diagnostic_target']},indent=2))
    if 'diagnostic_target' in result:print(json.dumps({k:v for k,v in result['diagnostic_target'].items() if k!='record'},indent=2))

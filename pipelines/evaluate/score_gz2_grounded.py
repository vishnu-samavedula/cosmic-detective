"""Score saved cloud predictions against frozen GZ2 labels; no model calls."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(run, split):
    ds = ROOT / 'datasets' / f'gz2_grounded_v1_{split}'
    manifest = json.loads((ROOT/'data/processed/gz2-grounded-v1/evaluation-manifests.json').read_text())[split]
    expected = manifest['samples']
    frame = pd.read_parquet(ROOT/'runs'/run/'results.parquet')
    assert len(frame) == len(expected)
    assert set(map(int, frame.sample_index)) == set(range(len(expected)))
    rows=[]
    for record in frame.to_dict('records'):
        i=int(record['sample_index']);ref=expected[i]
        messages=record['messages']
        if isinstance(messages,str):messages=json.loads(messages)
        raw=next(m['content'] for m in reversed(messages) if m['role']=='assistant')
        images=[x['image_url']['url'] for m in messages if m['role']=='user'
                for x in m['content'] if isinstance(x,dict) and x.get('type')=='image_url']
        assert len(images)==1
        digest=hashlib.sha256(base64.b64decode(images[0].split(',',1)[1])).hexdigest()
        assert digest==ref['image_sha256']
        text=raw.strip().lower().rstrip('.')
        strict=text if text in ('spiral','elliptical') else 'invalid'
        match=re.match(r'^the central galaxy(?: in the image)? is (spiral|elliptical|uncertain)\.',raw.strip().lower())
        extracted=strict if strict!='invalid' else match[1] if match else 'invalid'
        rows.append({'sample_index':i,'asset_id':ref['asset_id'],'object_id':ref['object_id'],
                     'reference':ref['training_label'],'prediction':strict,'explicit_classification':extracted,'raw_output':raw})
    rows.sort(key=lambda x:x['sample_index'])
    def summarize(key):
        classes={}
        for label in ['spiral','elliptical']:
            subset=[x for x in rows if x['reference']==label]
            correct=sum(x[key]==label for x in subset)
            classes[label]={'correct':correct,'total':len(subset),'recall':correct/len(subset)}
        correct=sum(x[key]==x['reference'] for x in rows)
        return {'correct':correct,'total':len(rows),'accuracy':correct/len(rows),
                'balanced_accuracy':sum(c['recall'] for c in classes.values())/2,
                'per_class':classes,'prediction_counts':dict(Counter(x[key] for x in rows)),
                'format_valid':sum(x['prediction']!='invalid' for x in rows)}
    result={'run':run,'split':split,'image_hashes_verified':len(rows),
            'strict':summarize('prediction'),'explicit_classification':summarize('explicit_classification'),
            'reference_note':'High-consensus GZ2 vote labels. Elliptical is a smooth appearance proxy. This sample is not representative of all GZ2.',
            'samples':rows}
    out=ROOT/'artifacts/evaluations/gz2-grounded-v1'/run
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--split',choices=['validation','test'],required=True)
    args=parser.parse_args()
    main(args.run,args.split)

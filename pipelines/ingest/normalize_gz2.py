"""Normalize all local GZ2 vote records and export a human-vote-grounded corpus.

Data processing only. No model weights, inference, or cloud submissions.
"""
import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from lqh.sources import image_folder

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/galaxy-zoo-2'
OUT = ROOT / 'data/processed/gz2-grounded-v1'
SEED = 20260912
QUESTION = ('Classify the central galaxy. Reply with exactly one word: spiral or elliptical. '
            'Spiral means visible spiral arms. Elliptical means a smooth elliptical-looking '
            'appearance, not a confirmed physical galaxy type.')
SYSTEM = 'You classify the visible morphology of the central galaxy. Follow the requested label format.\n'
PREFIX = 'gz2_grounded_v1'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalize():
    OUT.mkdir(parents=True, exist_ok=False)
    images = {x.path.stem: x for x in image_folder(RAW / 'kaggle-images/images_gz2/images', recursive=False)}
    mapping = pd.read_csv(RAW / 'gz2_filename_mapping.csv', dtype=str)
    votes = pd.read_csv(RAW / 'gz2_hart16.csv', dtype={'dr7objid': str})
    linked = mapping[mapping.asset_id.isin(images)].merge(votes, left_on='objid', right_on='dr7objid', validate='many_to_one')
    assert len(linked) == linked.asset_id.nunique() == linked.objid.nunique() == 239573
    branches = sorted({c.split('_', 1)[0] for c in votes if c.endswith('_count')})
    names = {}
    evidence = {}
    for branch in branches:
        count_columns = [c for c in votes if c.startswith(branch + '_') and c.endswith('_count')]
        n = linked[count_columns].sum(axis=1)
        evidence[f'{branch}.answers'] = n
        evidence[f'{branch}.answered'] = n.gt(0)
        for column in count_columns:
            stem = column[:-6]
            answer = re.sub(r'^t\d+_(.*?)_a\d+_', '', stem)
            for suffix in ['count', 'weight', 'fraction', 'weighted_fraction', 'debiased', 'flag']:
                source = stem + '_' + suffix
                if source not in linked: continue
                dest = f'{branch}.{answer}.{suffix}'
                values = linked[source]
                # Explicit null distinguishes unanswered questions from a no vote.
                if suffix in ['fraction', 'weighted_fraction', 'debiased', 'flag']:
                    values = values.where(n.gt(0))
                evidence[dest] = values
                names[dest] = source
    frame = pd.DataFrame(evidence)
    frame.insert(0, 'object_id', linked.objid)
    frame.insert(1, 'asset_id', linked.asset_id)
    frame.insert(2, 'image_path', linked.asset_id.map(lambda x: str(images[x].path.relative_to(ROOT))))
    for field in ['ra', 'dec', 'gz2_class', 'total_classifications', 'total_votes']:
        frame[field] = linked[field]
    def agrees(branch, answer, threshold, count):
        n = frame[f'{branch}.answers']
        return (n >= count) & (frame[f'{branch}.{answer}.count'] / n >= threshold) & (frame[f'{branch}.{answer}.weighted_fraction'] >= threshold)
    ordinary = agrees('t06', 'no', .8, 20)
    smooth = agrees('t01', 'smooth', .9, 20) & ordinary
    spiral = agrees('t01', 'features_or_disk', .8, 20) & agrees('t02', 'no', .8, 10) & agrees('t04', 'spiral', .8, 10) & ordinary
    frame['training_label'] = pd.Series(pd.NA, index=frame.index, dtype='string')
    frame.loc[smooth, 'training_label'] = 'elliptical'
    frame.loc[spiral, 'training_label'] = 'spiral'
    frame['label_status'] = 'not_selected_by_binary_consensus_policy'
    frame.loc[smooth | spiral, 'label_status'] = 'strong_human_vote_support'
    frame['irregular_feature_supported'] = agrees('t06', 'yes', .8, 20) & agrees('t08', 'irregular', .7, 10)
    frame['star_or_artifact_supported'] = agrees('t01', 'star_or_artifact', .8, 20)
    frame['edge_on_supported'] = agrees('t01', 'features_or_disk', .8, 20) & agrees('t02', 'yes', .8, 10)
    frame.to_parquet(OUT / 'catalog.parquet', index=False)
    frame[['object_id', 'asset_id', 'image_path', 'gz2_class', 'training_label', 'label_status', 'irregular_feature_supported', 'star_or_artifact_supported', 'edge_on_supported']].to_csv(OUT / 'labels.csv', index=False)
    unmatched = mapping[mapping.asset_id.isin(set(images) - set(frame.asset_id))].copy()
    unmatched['image_path'] = unmatched.asset_id.map(lambda x: str(images[x].path.relative_to(ROOT)))
    assert unmatched.asset_id.nunique() == 3861
    unmatched.to_csv(OUT / 'unmatched-images.csv', index=False)
    summary = {'normalization_complete': True, 'images': len(images), 'matched': len(frame), 'unmatched_images': len(unmatched),
               'strong_binary_candidates': frame.training_label.value_counts().to_dict(),
               'irregular_feature_candidates': int(frame.irregular_feature_supported.sum()),
               'no_binary_target': int(frame.training_label.isna().sum()),
               'policy': {'smooth': 'raw and weighted >=0.90, >=20 answers to t01', 'spiral': 'features/disk >=0.80 with >=20 t01 answers; not-edge-on and spiral >=0.80 with >=10 answers each; both raw and weighted', 'both_classes': 'nothing odd >=0.80 raw and weighted, >=20 t06 answers', 'irregular': 'odd yes >=0.80 (>=20 t06); irregular >=0.70 (>=10 t08), raw and weighted', 'elliptical_semantics': 'Smooth/elliptical-looking proxy; may include S0. Not confirmed physical type.', 'missing': 'Unanswered branch fractions/flags normalized to null. Counts retained. No automatic uncertain target.', 'synthetic_relabeling': False},
               'sources': {str(p.relative_to(ROOT)): sha(p) for p in [RAW/'gz2_hart16.csv', RAW/'gz2_filename_mapping.csv']}}
    write_json(OUT / 'column-mapping.json', names)
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def export():
    assert not (OUT / 'splits.parquet').exists(), 'Refusing to overwrite frozen splits'
    catalog = pd.read_parquet(OUT / 'catalog.parquet')
    # Exclude identities/hashes appearing in previous experiment provenance or reviews.
    prior_paths = list((ROOT/'datasets').glob('*/manifest.json')) + list((ROOT/'data/splits').glob('*.json'))
    prior_paths += list((ROOT/'artifacts/evaluations').rglob('*.json'))
    prior_paths += list((ROOT/'seed_data').rglob('*.json')) + list((ROOT/'seed_data').rglob('*.jsonl'))
    prior_paths += [ROOT/'apps/web/client/public/catalog.json']
    old_ids, old_hashes = set(), set()
    for p in prior_paths:
        if not p.is_file(): continue
        text = p.read_text()
        old_ids.update(re.findall(r'(?<!\d)\d{18}(?!\d)', text))
        old_hashes.update(re.findall(r'\b[0-9a-f]{64}\b', text))
    candidates = catalog[catalog.training_label.notna() & ~catalog.object_id.isin(old_ids)].sample(frac=1, random_state=SEED)
    seen = set(old_hashes)
    rows, rejected = [], Counter()
    for i, row in enumerate(candidates.to_dict('records')):
        path = ROOT / row['image_path']
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            rejected['duplicate_or_previous_image_hash'] += 1; continue
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except Exception:
            rejected['invalid_image'] += 1; continue
        seen.add(digest)
        rows.append({k: row[k] for k in ['object_id', 'asset_id', 'image_path', 'training_label', 'gz2_class']} | {'image_sha256': digest})
        if i % 5000 == 0: print(f'Checked {i}/{len(candidates)} eligible images', flush=True)
    split = pd.DataFrame(rows)
    split['split'] = 'train'
    for label in ['spiral', 'elliptical']:
        indices = list(split.index[split.training_label.eq(label)])
        assert len(indices) > 1000
        split.loc[indices[:200], 'split'] = 'test'
        split.loc[indices[200:400], 'split'] = 'validation'
    assert split.object_id.is_unique and split.image_sha256.is_unique
    split.to_parquet(OUT/'splits.parquet', index=False)
    split.to_csv(OUT/'splits.csv', index=False)
    groups = {'train_spiral': split[(split.split=='train') & (split.training_label=='spiral')],
              'train_elliptical': split[(split.split=='train') & (split.training_label=='elliptical')],
              'validation': split[split.split=='validation'], 'test': split[split.split=='test']}
    schema = pa.schema([('messages', pa.string()), ('audio', pa.null()), ('tools', pa.null())])
    dataset_info = {}
    for name, group in groups.items():
        dest = ROOT/'datasets'/f'{PREFIX}_{name}'
        dest.mkdir(exist_ok=False)
        # Stream batches so the full image corpus is never duplicated in RAM.
        with pq.ParquetWriter(dest/'data.parquet', schema, compression='zstd') as writer:
            batch = []
            for sample in group.to_dict('records'):
                raw = (ROOT/sample['image_path']).read_bytes()
                messages = [{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}}, {'type':'text','text':QUESTION}]}, {'role':'assistant','content':sample['training_label']}]
                batch.append({'messages':json.dumps(messages), 'audio':None, 'tools':None})
                if len(batch) == 256:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema)); batch=[]
            if batch: writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        info = {'name':dest.name, 'rows':len(group), 'purpose':'Human-vote-grounded binary morphology '+name, 'label_source':'data/processed/gz2-grounded-v1/catalog.parquet', 'samples':group.to_dict('records'), 'image_bytes':'unchanged original JPEG', 'student_input':'image plus task instruction; no IDs, votes, or labels', 'assistant_target':'one word', 'question':QUESTION, 'data_sha256':sha(dest/'data.parquet')}
        write_json(dest/'manifest.json', info)
        dataset_info[name] = {'path':str(dest.relative_to(ROOT)), 'rows':len(group), 'bytes':(dest/'data.parquet').stat().st_size, 'sha256':info['data_sha256']}
        print(f'Exported {name}: {len(group)} rows', flush=True)
    audit = {'seed':SEED, 'prior_object_ids_excluded':len(old_ids & set(catalog.object_id)), 'rejected_images':dict(rejected), 'unique_images':len(split), 'datasets':dataset_info, 'class_counts':split.groupby(['split','training_label']).size().to_dict()}
    audit['class_counts'] = {f'{a}/{b}':int(v) for (a,b),v in audit['class_counts'].items()}
    write_json(OUT/'export-audit.json', audit)
    # Review only training images; test answers remain outside manual review.
    review = pd.concat([groups['train_spiral'].head(12),groups['train_elliptical'].head(12)])
    cards=[]
    for row in review.to_dict('records'):
        image='data:image/jpeg;base64,'+base64.b64encode((ROOT/row['image_path']).read_bytes()).decode()
        cards.append(f'<article><img src="{image}"><h3>{html.escape(row["training_label"])} · {row["asset_id"]}</h3><p>GZ2: {html.escape(row["gz2_class"])}</p></article>')
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>GZ2 grounded training review</title><style>body{font:16px system-ui;background:#101418;color:#eee;padding:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:20px}img{width:220px;height:220px;object-fit:contain}</style><h1>24 training examples</h1><p>Targets come from strong human votes. Elliptical is a smooth appearance proxy. No test images shown.</p><div class="grid">'+''.join(cards)+'</div>')
    (ROOT/'prompts/gz2_grounded_v1.md').write_text(SYSTEM)
    print(json.dumps(audit,indent=2),flush=True)


def export_unmatched():
    records = pd.read_csv(OUT/'unmatched-images.csv', dtype=str).to_dict('records')
    dest = ROOT/'datasets'/f'{PREFIX}_unmatched'
    dest.mkdir(exist_ok=False)
    samples, batch = [], []
    schema = pa.schema([('messages',pa.string()),('audio',pa.null()),('tools',pa.null())])
    with pq.ParquetWriter(dest/'data.parquet',schema,compression='zstd') as writer:
        for i, record in enumerate(records):
            raw=(ROOT/record['image_path']).read_bytes()
            with Image.open(io.BytesIO(raw)) as image: image.verify()
            messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}},{'type':'text','text':QUESTION}]}]
            batch.append({'messages':json.dumps(messages),'audio':None,'tools':None})
            samples.append({'sample_index':i,'object_id':record['objid'],'asset_id':record['asset_id'],'image_path':record['image_path'],'image_sha256':hashlib.sha256(raw).hexdigest(),'reference':None})
            if len(batch)==256:
                writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
        if batch: writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    assert len(samples)==3861
    write_json(dest/'manifest.json',{'name':dest.name,'rows':len(samples),'purpose':'Inference only; no matching Hart labels and no assistant reference targets','question':QUESTION,'samples':samples,'prediction_limits':'Binary closed-set predictions; no accuracy or out-of-scope detection guarantee','data_sha256':sha(dest/'data.parquet')})
    print(f'Prepared {len(samples)} unmatched images for cloud inference; no reference targets.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['normalize','export','export_unmatched'])
    args=parser.parse_args()
    {'normalize':normalize,'export':export,'export_unmatched':export_unmatched}[args.action]()

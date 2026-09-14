"""Data-only export of a balanced subset of the frozen GZ2 training split."""
import base64
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from normalize_gz2 import QUESTION, ROOT, sha, write_json
from lqh.lineage.datamix import DatamixEntry, stream_hashes


def main():
    name = 'gz2_grounded_12k_v1'
    dest = ROOT / 'datasets' / name
    audit_dir = ROOT / 'training/lqh/gz2-grounded-12k-v1'
    if dest.exists() or audit_dir.exists():
        raise FileExistsError('Refusing to overwrite an existing export or audit')
    splits_path = ROOT / 'data/processed/gz2-grounded-v1/splits.parquet'
    splits = pd.read_parquet(splits_path)
    train = splits[splits.split.eq('train')].copy()
    seed = 'gz2-balanced-12k-v1:20260911'
    train['selection_rank'] = train.object_id.map(
        lambda oid: hashlib.sha256(f'{seed}:{oid}'.encode()).hexdigest()
    )
    selected = pd.concat([
        train[train.training_label.eq(label)].sort_values('selection_rank').head(6000)
        for label in ('elliptical', 'spiral')
    ]).sort_values('selection_rank').reset_index(drop=True)
    assert selected.training_label.value_counts().to_dict() == {'elliptical': 6000, 'spiral': 6000}
    assert selected.object_id.nunique() == selected.image_sha256.nunique() == 12000
    held = splits[splits.split.isin(['validation', 'test'])]
    assert not set(selected.object_id) & set(held.object_id)
    assert not set(selected.image_sha256) & set(held.image_sha256)
    frozen_paths = [ROOT / 'datasets' / f'gz2_grounded_v1_{part}' / 'data.parquet'
                    for part in ('validation', 'test')]
    frozen_hashes = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    dest.mkdir()
    audit_dir.mkdir(parents=True)
    selected.to_csv(audit_dir / 'membership.csv', index=False)
    schema = pa.schema([('messages', pa.string()), ('audio', pa.null()), ('tools', pa.null())])
    batch = []
    with pq.ParquetWriter(dest / 'data.parquet', schema, compression='zstd') as writer:
        for i, row in enumerate(selected.to_dict('records'), 1):
            raw = (ROOT / row['image_path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
            with Image.open(io.BytesIO(raw)) as img:
                img.verify()
            messages = [
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}},
                    {'type': 'text', 'text': QUESTION},
                ]},
                {'role': 'assistant', 'content': row['training_label']},
            ]
            batch.append({'messages': json.dumps(messages), 'audio': None, 'tools': None})
            if len(batch) == 256:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch = []
            if i % 2000 == 0:
                print(f'Exported and image-verified {i:,}/12,000', flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    # Read the exported bytes back and verify every image, input and target.
    index = 0
    for part in pq.ParquetFile(dest / 'data.parquet').iter_batches(batch_size=256):
        for row in part.to_pylist():
            messages = json.loads(row['messages'])
            expected = selected.iloc[index]
            assert len(messages) == 2
            assert messages[0]['role'] == 'user'
            assert messages[0]['content'][1] == {'type': 'text', 'text': QUESTION}
            assert messages[1] == {'role': 'assistant', 'content': expected.training_label}
            raw = base64.b64decode(messages[0]['content'][0]['image_url']['url'].split(',', 1)[1], validate=True)
            assert hashlib.sha256(raw).hexdigest() == expected.image_sha256
            index += 1
    assert index == 12000
    assert frozen_hashes == {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    manifest = {
        'name': name, 'purpose': 'training', 'rows': 12000,
        'provenance': {'kind': 'filtered', 'selection': '6000 per class from frozen training split', 'selection_seed': seed},
        'derived_from': 'datasets/gz2_grounded_v1_train_elliptical',
        'parent_dataset': 'datasets/gz2_grounded_v1_train_spiral',
        'sources': [{'path': f'datasets/gz2_grounded_v1_train_{label}'} for label in ('elliptical', 'spiral')],
        'membership': str((audit_dir / 'membership.csv').relative_to(ROOT)),
        'data_sha256': sha(dest / 'data.parquet'),
    }
    write_json(dest / 'manifest.json', manifest)
    write_json(audit_dir / 'manifest.snapshot.json', manifest)
    # Actual LQH serializer, with UUID-length placeholders until registration.
    placeholder = '00000000-0000-4000-8000-000000000000'
    summary = stream_hashes(ROOT, [DatamixEntry(path=str(dest.relative_to(ROOT)), dataset_object_id=placeholder)])
    payload = {'datamix_version_id': placeholder, 'stream_sha': summary.sha,
               'rows': [r.to_api() for r in summary.unique]}
    import httpx
    body_bytes = len(httpx.Request('POST', 'https://example.invalid', json=payload).content)
    assert len(summary.unique) == summary.n_rows == 12000
    assert body_bytes < 8 * 2**20
    audit = {
        'rows': index, 'class_counts': selected.training_label.value_counts().to_dict(),
        'original_split': 'train', 'object_overlap_with_heldout': 0, 'image_overlap_with_heldout': 0,
        'all_image_bytes_and_labels_verified': True, 'heldout_files_unchanged': frozen_hashes,
        'selection_seed': seed, 'source_splits_sha256': sha(splits_path),
        'parquet_bytes': (dest / 'data.parquet').stat().st_size, 'parquet_sha256': manifest['data_sha256'],
        'hygiene_request_bytes': body_bytes, 'request_limit_bytes': 8 * 2**20,
        'payload_uuid_note': 'UUID-length placeholders; same encoded size as registration UUIDs',
        'stream_sha256': summary.sha,
    }
    write_json(audit_dir / 'verification.json', audit)
    write_json(audit_dir / 'register.args.json', {'path': str(dest.relative_to(ROOT)), 'reason': 'User-approved balanced 12000-image subset of frozen training split; preserve both source lineages', 'purpose': 'training', 'provenance_kind': 'filtered'})
    write_json(audit_dir / 'datamix.args.json', {'entries': [{'dataset': str(dest.relative_to(ROOT)), 'repeat': 1, 'role': 'train'}]})
    args = json.loads((ROOT / 'training/lqh/gz2-grounded-v1/start-training.args.json').read_text())
    args.update(dataset=str(dest.relative_to(ROOT)), run_name='sft_gz2_grounded_450m_12k_v1')
    write_json(audit_dir / 'start-training.args.json', args)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()

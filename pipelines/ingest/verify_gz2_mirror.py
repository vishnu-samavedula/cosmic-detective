#!/usr/bin/env python3
"""Verify the Kaggle mirror and extract a deterministic 200-object development sample."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/galaxy-zoo-2'
OUT = ROOT / 'data/processed/galaxy-zoo-2/fixture'
REPORT = ROOT / 'data/manifests/galaxy-zoo-2.mirror-report.json'


def main():
    path = RAW / 'kaggle-images.zip'
    if not path.exists():
        path = RAW / 'kaggle-images.zip.part'
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Corrupt archive member: {bad}')
        names = archive.namelist()
        mappings = [n for n in names if Path(n).name == 'gz2_filename_mapping.csv']
        if len(mappings) != 1:
            raise ValueError('Expected exactly one mapping table')
        mapping_bytes = archive.read(mappings[0])
        md5 = hashlib.md5(mapping_bytes).hexdigest()
        if md5 != '7e28465e6dfbf96c0d828c595a5bbd80':
            raise ValueError(f'Mapping differs from original release: {md5}')
        images = {}
        for n in names:
            if n.lower().endswith('.jpg'):
                asset = Path(n).stem
                if asset in images:
                    raise ValueError(f'Duplicate image asset: {asset}')
                images[asset] = n
        label_csv = RAW / 'gz2_hart16.csv'
        label_stream = label_csv.open('rt', newline='') if label_csv.exists() else gzip.open(RAW / 'gz2_hart16.csv.gz', 'rt', newline='')
        with label_stream as stream:
            reader = csv.DictReader(stream)
            labels = {row['dr7objid']: row for row in reader}
        rows = list(csv.DictReader(io.StringIO(mapping_bytes.decode('utf-8-sig'))))
        matched, seen = [], set()
        missing_image = missing_label = duplicate_object = 0
        for row in rows:
            obj, asset = row['objid'], row['asset_id']
            if asset not in images:
                missing_image += 1
                continue
            if obj not in labels:
                missing_label += 1
                continue
            if obj in seen:
                duplicate_object += 1
                continue
            seen.add(obj)
            matched.append((obj, asset))
        if len(matched) < 200:
            raise ValueError('Fewer than 200 matched objects')
        OUT.mkdir(parents=True, exist_ok=True)
        selected = random.Random(42).sample(sorted(matched), 200)
        records = []
        for obj, asset in selected:
            content = archive.read(images[asset])
            if not content.startswith(b'\xff\xd8\xff'):
                raise ValueError(f'Not JPEG: {asset}')
            (OUT / f'{obj}.jpg').write_bytes(content)
            records.append({'object_id': obj, 'asset_id': asset,
                            'image': f'{obj}.jpg', 'ra': labels[obj]['ra'],
                            'dec': labels[obj]['dec'],
                            'image_sha256': hashlib.sha256(content).hexdigest()})
        (OUT / 'objects.json').write_text(json.dumps(records, indent=2) + '\n')
        # Kept outside public asset records. This fixture is development-only.
        (OUT / 'labels.private.json').write_text(json.dumps(
            {obj: labels[obj] for obj, _ in selected}, indent=2) + '\n')
    (RAW / 'gz2_filename_mapping.csv').write_bytes(mapping_bytes)
    final = RAW / 'kaggle-images.zip'
    if path != final:
        path.replace(final)
    report = {
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'source': 'https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images',
        'archive_sha256': sha.hexdigest(), 'archive_bytes': final.stat().st_size,
        'zip_crc_verified': True, 'mapping_published_md5_verified': True,
        'mapping_rows': len(rows), 'jpeg_count': len(images),
        'matched_unique_objects': len(matched), 'mapping_rows_missing_image': missing_image,
        'image_rows_missing_hart_label': missing_label,
        'duplicate_matched_objects': duplicate_object,
        'development_fixture_count': len(records), 'fixture_seed': 42,
        'limitations': ['Mirror is repackaged: original image ZIP checksum cannot be compared.',
                       'JPEG signatures and ZIP integrity checked; full image decoding and visual join review pending.',
                       'Fixture is a random development sample, not a balanced training set or held-out test set.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

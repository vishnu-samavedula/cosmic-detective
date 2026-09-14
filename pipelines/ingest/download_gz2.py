#!/usr/bin/env python3
"""Download original sources with resumable transfers and integrity checks."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/manifests/galaxy-zoo-2.sources.json'
RAW = ROOT / 'data/raw/galaxy-zoo-2'
REPORT = ROOT / 'data/manifests/galaxy-zoo-2.download-report.json'


def validate(path, spec):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    if spec['published_md5'] and md5.hexdigest() != spec['published_md5']:
        raise ValueError(f"Published checksum mismatch: {spec['name']}")
    details = {}
    if spec['name'].endswith('.zip'):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f'Corrupt ZIP member: {bad}')
            details['jpeg_count'] = sum(n.lower().endswith('.jpg') for n in archive.namelist())
    elif spec['name'].endswith('.csv.gz'):
        with gzip.open(path, 'rt', newline='') as stream:
            reader = csv.reader(stream)
            details['columns'] = next(reader)
            details['row_count'] = sum(1 for _ in reader)
        if 'dr7objid' not in details['columns']:
            raise ValueError('Expected Hart classification object ID column absent')
    elif spec['name'].endswith('.csv'):
        with path.open(newline='') as stream:
            reader = csv.reader(stream)
            details['columns'] = next(reader)
            details['row_count'] = sum(1 for _ in reader)
        if not {'objid', 'asset_id', 'sample'}.issubset(details['columns']):
            raise ValueError('Expected filename mapping columns absent')
    return dict(status='verified', bytes=path.stat().st_size,
                sha256=sha.hexdigest(), md5=md5.hexdigest(), **details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', help='Download selected manifest filenames only')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    names = {s['name'] for s in manifest['files']}
    if args.only and set(args.only) - names:
        parser.error('Unknown filename in --only')
    RAW.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {'files': {}}
    failures = 0
    for spec in manifest['files']:
        name = spec['name']
        if args.only and name not in args.only:
            continue
        dest = RAW / name
        partial = RAW / (name + '.part')
        print(f'Processing {name}', flush=True)
        try:
            if not dest.exists():
                if args.verify_only:
                    raise FileNotFoundError(f'Not downloaded: {name}')
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                '--connect-timeout', '20', '--max-time', '3600',
                                '--speed-limit', '1024', '--speed-time', '60',
                                '--retry', '2', '--retry-delay', '5', '--continue-at', '-',
                                '--output', str(partial), spec['url']], check=True)
                result = validate(partial, spec)
                partial.replace(dest)
            else:
                result = validate(dest, spec)
            report['files'][name] = dict(url=spec['url'], **result)
            print(f"Verified {name}: {result['bytes']:,} bytes", flush=True)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
            failures += 1
            report['files'][name] = dict(url=spec['url'], status='failed', error=str(exc))
            print(f'FAILED {name}: {exc}', flush=True)
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

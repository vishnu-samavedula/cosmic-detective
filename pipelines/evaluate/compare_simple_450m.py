"""Analyze saved LQH cloud predictions; never loads models or submits jobs."""
import argparse
import base64
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd

from simple_morphology_zero_shot import read_outputs, summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/evaluations/simple-morphology-450m-v1'
RUN = 'baseline_simple_morphology_450m_v1_r1'
BASE = 'baseline_simple_morphology_3b_v1_r1'


def main(run_name=RUN, output_dir=OUT, model_label='450M'):
    RUN, OUT = run_name, Path(output_dir)
    model_runs = [('3B', BASE)]
    if RUN != 'baseline_simple_morphology_450m_v1_r1':
        model_runs.append(('450M', 'baseline_simple_morphology_450m_v1_r1'))
    model_runs.append((model_label, RUN))
    run_names = [run for _, run in model_runs]
    frozen = json.loads((OUT / 'input-integrity.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
               for p, h in frozen.items()), 'Baseline inputs changed'
    manifest = json.loads((ROOT / 'datasets/galaxy_simple_morphology_v1_test/manifest.json').read_text())
    samples = manifest['samples']
    configs = [json.loads((ROOT / 'runs' / run / 'config.json').read_text()) for run in run_names]
    protocol = ['training_method', 'eval_dataset', 'scorer', 'judge_size',
                'max_new_tokens', 'num_samples', 'system_prompt']
    assert all(configs[0][k] == config[k] for config in configs[1:] for k in protocol)
    outputs = {run: read_outputs(run) for run in run_names}
    expected_indices = {s['sample_index'] for s in samples}
    assert len(expected_indices) == 100
    for run, values in outputs.items():
        assert set(values) == expected_indices, run
    image_checks = 0
    for row in pd.read_parquet(ROOT / 'runs' / RUN / 'results.parquet').to_dict('records'):
        msgs = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
        images = [c['image_url']['url'] for m in msgs if m['role'] == 'user'
                  for c in m['content'] if isinstance(c, dict) and c.get('type') == 'image_url']
        assert len(images) == 1
        sample = samples[int(row['sample_index'])]
        assert hashlib.sha256(base64.b64decode(images[0].split(',', 1)[1])).hexdigest() == sample['image_sha256']
        image_checks += 1
    def parse(raw):
        label = raw.strip().lower().rstrip('.')
        return label if label in ('spiral', 'elliptical', 'uncertain') else 'invalid'
    predictions = {run: {i: parse(raw) for i, raw in values.items()} for run, values in outputs.items()}
    summaries = {run: summarize(samples, predictions[run]) for run in run_names}
    # Secondary diagnostic: accept only an explicit initial classification.
    # Keep the original label-only metric unchanged and apply this rule to all runs.
    def explicit_label(raw):
        text = raw.strip().lower()
        match = re.fullmatch(r'(spiral|elliptical|uncertain)\.?', text)
        if not match:
            match = re.match(r'^the central galaxy(?: in the image)? is (spiral|elliptical|uncertain)\.', text)
        return match[1] if match else 'invalid'
    extracted = {run: {i: explicit_label(raw) for i, raw in values.items()} for run, values in outputs.items()}
    extracted_summaries = {run: summarize(samples, extracted[run]) for run in run_names}
    per_image = []
    cards = []
    for s in samples:
        i = s['sample_index']
        ref = s['simple_reference']
        entry = {'sample_index': i, 'asset_id': s['asset_id'], 'reference': ref,
                 'category': s['category'], 'predictions': {r: predictions[r][i] for r in run_names},
                 'explicit_classifications': {r: extracted[r][i] for r in run_names},
                 'raw_outputs': {r: outputs[r][i] for r in run_names}}
        per_image.append(entry)
        image_path = ROOT / f"data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images/{s['asset_id']}.jpg"
        uri = 'data:image/jpeg;base64,' + base64.b64encode(image_path.read_bytes()).decode()
        labels = ' · '.join(f'{html.escape(name)}: {extracted[run][i]} (strict: {predictions[run][i]})' for name, run in model_runs)
        cards.append(f'<article><img src="{uri}" alt="Galaxy {html.escape(str(s["asset_id"]))}"><div>'
                     f'<h2>{html.escape(str(s["asset_id"]))}.jpg</h2><p>Reference: {html.escape(ref or "unscored")}</p>'
                     f'<p>{labels}</p>'
                     f'<details><summary>{html.escape(model_label)} raw output</summary><pre>{html.escape(outputs[RUN][i])}</pre></details></div></article>')
    result = {'protocol': 'Same 100 images, prompt, 32-token limit, unconstrained decoding and deterministic label scoring; LQH cloud.',
              'reference_policy': manifest['reference_policy'],
              'revision_note': '450M and 1.6B runs request main through LQH; no independently pinned commit. 3B uses its recorded pinned revision.',
              'explicit_classification_policy': 'Post-hoc diagnostic applied to every model: accept a bare allowed label, or an opening sentence The central galaxy [in the image] is <label>. Do not infer a label from arbitrary prose. Primary strict metric remains unchanged.',
              'explicit_classification_summaries': extracted_summaries,
              'verified_image_hashes': image_checks, 'summaries': summaries, 'samples': per_image}
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Simple zero-shot morphology: model comparison', '', result['protocol'], '', '## Original strict label-only scoring', '',
             '| Model | Correct / 58 | Spiral / 28 | Smooth proxy / 30 | Invalid / 100 |',
             '| --- | --- | --- | --- | --- |']
    for name, run in model_runs:
        s = summaries[run]
        lines.append(f'| {name} | {s["correct"]}/58 ({s["accuracy"]:.1%}) | {s["per_class"]["spiral"]["correct"]}/28 | {s["per_class"]["elliptical"]["correct"]}/30 | {s["all100_predictions"].get("invalid", 0)} |')
    lines += ['', '## Explicit classification extracted from the response', '',
              result['explicit_classification_policy'], '',
              '| Model | Correct / 58 | Spiral / 28 | Smooth proxy / 30 |',
              '| --- | --- | --- | --- |']
    for name, run in model_runs:
        s = extracted_summaries[run]
        lines.append(f'| {name} | {s["correct"]}/58 ({s["accuracy"]:.1%}) | {s["per_class"]["spiral"]["correct"]}/28 | {s["per_class"]["elliptical"]["correct"]}/30 |')
    lines += ['', 'Uncertain/invalid answers count as incorrect on the 58 supported references. The other 42 images have no asserted binary reference and are excluded from accuracy. Smooth is an elliptical-looking proxy, not a confirmed physical type.', '',
              'This is a development comparison on a repeatedly used sample, not a fresh final benchmark. Model families differ in architecture/pretraining as well as size.', '', result['revision_note'], '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))
    (OUT / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Model comparison</title><style>body{background:#121417;color:#eee;font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px}article{display:flex;gap:24px;border-top:1px solid #555;padding:20px 0}img{width:200px;height:200px;object-fit:contain}pre{white-space:pre-wrap}</style><h1>Model comparison — zero-shot</h1><p>58 supported references out of 100 images. Elliptical means a smooth appearance proxy.</p>' + ''.join(cards))
    print(json.dumps({'strict': summaries, 'explicit_classification': extracted_summaries}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--label', default='450M')
    args = parser.parse_args()
    main(args.run, args.output_dir, args.label)

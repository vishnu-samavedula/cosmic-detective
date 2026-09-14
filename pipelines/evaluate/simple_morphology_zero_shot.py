"""Prepare/analyze a simple cloud VLM test; local work is data processing only."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/galaxy_simple_morphology_v1_test"
OUT = ROOT / "artifacts/evaluations/simple-morphology-v1"
RUN = "baseline_simple_morphology_3b_v1_r1"
SYSTEM = """Classify the visible morphology of the central galaxy in the image.
Reply with exactly one label: spiral, elliptical, or uncertain.
Use spiral when spiral arms are visible. Use elliptical for a smooth, rounded or elongated elliptical-looking galaxy without visible spiral structure. Use uncertain when the image is unclear, the galaxy is edge-on, or neither description fits. Judge only what is visible; do not guess hidden structures.
"""
QUESTION = "Is the central galaxy spiral, elliptical, or uncertain?"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_label(features):
    if features["appearance"] == "smooth":
        return "elliptical"
    if features["spiral"] == "yes":
        return "spiral"
    return None


def old_prediction(raw):
    """Project relevant old JSON fields; unrelated fields do not affect this task."""
    try:
        f = json.loads(raw)["features"]
        if f.get("appearance") == "smooth":
            if f.get("spiral") is not None or f.get("edge_on") is not None:
                return "invalid"
            return "elliptical"
        if f.get("appearance") == "features_or_disk":
            if f.get("edge_on") == "no" and f.get("spiral") == "yes":
                return "spiral"
            if f.get("spiral") == "yes":
                return "invalid"
        elif f.get("spiral") is not None:
            return "invalid"
        return "uncertain"
    except (ValueError, KeyError, TypeError):
        return "invalid"


def prepare():
    source = ROOT / "datasets/galaxy_json_pilot_v1_test"
    original = json.loads((source / "manifest.json").read_text())
    data = pd.read_parquet(source / "data.parquet")
    frozen = [source / "manifest.json", source / "data.parquet"]
    for split in ("train", "validation"):
        path = ROOT / f"datasets/galaxy_json_pilot_v1_{split}_raw/manifest.json"
        other = json.loads(path.read_text())
        assert not ({s["object_id"] for s in original["samples"]} &
                    {s["object_id"] for s in other["samples"]})
        assert not ({s["image_sha256"] for s in original["samples"]} &
                    {s["image_sha256"] for s in other["samples"]})
        frozen.append(path)
    assert len(data) == len(original["samples"]) == 100
    records, samples = [], []
    for i, sample in enumerate(original["samples"]):
        assert sample["sample_index"] == i
        messages = json.loads(data.iloc[i]["messages"])
        image = [part for msg in messages if msg["role"] == "user"
                 for part in msg["content"] if part["type"] == "image_url"]
        assert len(image) == 1
        raw = base64.b64decode(image[0]["image_url"]["url"].split(",", 1)[1])
        assert hashlib.sha256(raw).hexdigest() == sample["image_sha256"]
        label = reference_label(sample["reference"]["features"])
        records.append({"messages": json.dumps([
            {"role": "user", "content": image + [{"type": "text", "text": QUESTION}]},
            {"role": "assistant", "content": label or "uncertain"}], ensure_ascii=False),
            "audio": None, "tools": None})
        samples.append({**sample, "simple_reference": label,
                        "scored_binary_proxy": label is not None})
    DATASET.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(DATASET / "data.parquet", index=False)
    manifest = {
        "source_dataset": str(source.relative_to(ROOT)),
        "protocol": "All 100 original held-out images, identical JPEG bytes/order. New short prompt; no examples, identity, votes or reference labels in inference input. Unconstrained plain-label decoding.",
        "reference_policy": "30 smooth galaxies serve as elliptical-looking proxies, 28 branch-supported spiral=yes galaxies as spiral proxies. Same >=10 votes / >=70% raw conditional agreement rule as prior pilots. Smooth is not confirmed physical elliptical type. Other 42 images are excluded from binary accuracy; their response distribution is reported, with no assumed correct class. Their stored assistant target 'uncertain' is only a dataset placeholder stripped before inference.",
        "system_prompt": SYSTEM, "user_prompt": QUESTION, "samples": samples,
        "frozen_source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in frozen},
    }
    (DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "prompts/galaxy_simple_morphology_v1.md").write_text(SYSTEM)
    (ROOT / "evals/scorers/galaxy_simple_morphology_v1.md").write_text(
        "Assess the output label against the central galaxy image. Allowed labels are spiral, elliptical, uncertain. "
        "Spiral means visible spiral arms; elliptical is only a smooth elliptical-looking appearance, not a physical identity claim. "
        "Uncertain is appropriate for edge-on, unclear or neither appearance. Return score 1-10 and brief reasoning. "
        "This visual judge is advisory only. A separate deterministic analysis compares labels to held-out volunteer-derived references; it supplies the primary metric.\n")
    args = {"repo": "LiquidAI/LFM2.5-VL-3B", "revision": "a3af5799199acdd2a4f56ac4342816abb46c12a9",
            "training_method": "full", "eval_dataset": str(DATASET.relative_to(ROOT)),
            "scorer": "evals/scorers/galaxy_simple_morphology_v1.md", "judge_size": "small",
            "system_prompt_path": "prompts/galaxy_simple_morphology_v1.md",
            "max_new_tokens": 32, "timeout_minutes": 15, "run_name": RUN}
    (OUT / "eval.args.json").write_text(json.dumps(args, indent=2) + "\n")
    print(json.dumps({"images": len(samples), "references": dict(Counter(s["simple_reference"] or "unscored" for s in samples)), "args": args}, indent=2))


def read_outputs(run):
    df = pd.read_parquet(ROOT / "runs" / run / "results.parquet")
    out = {}
    for _, row in df.iterrows():
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        out[int(row["sample_index"])] = next(m["content"] for m in reversed(messages) if m["role"] == "assistant")
    return out


def summarize(samples, predictions):
    scored = [s for s in samples if s["scored_binary_proxy"]]
    per_class = {}
    for label in ("spiral", "elliptical"):
        group = [s for s in scored if s["simple_reference"] == label]
        correct = sum(predictions.get(s["sample_index"], "missing") == label for s in group)
        per_class[label] = {"correct": correct, "total": len(group), "recall": correct / len(group)}
    correct = sum(v["correct"] for v in per_class.values())
    answered = sum(predictions.get(s["sample_index"]) in ("spiral", "elliptical") for s in scored)
    return {
        "correct": correct, "scored": len(scored), "accuracy": correct / len(scored),
        "balanced_accuracy": sum(x["recall"] for x in per_class.values()) / 2,
        "per_class": per_class, "binary_answer_coverage": answered / len(scored),
        "accuracy_when_binary_answered": correct / answered if answered else None,
        "all100_predictions": dict(Counter(predictions.get(s["sample_index"], "missing") for s in samples)),
        "unscored42_predictions": dict(Counter(predictions.get(s["sample_index"], "missing") for s in samples if not s["scored_binary_proxy"])),
        "confusion": {label: dict(Counter(predictions.get(s["sample_index"], "missing") for s in scored if s["simple_reference"] == label)) for label in per_class},
        "by_original_category": {category: dict(Counter(predictions.get(s["sample_index"], "missing") for s in samples if s["category"] == category)) for category in sorted({s["category"] for s in samples})},
    }


def analyze():
    manifest = json.loads((DATASET / "manifest.json").read_text())
    samples = manifest["samples"]
    comparisons = [
        ("Base 3B — simple zero-shot prompt", RUN, False),
        ("Base 3B — previous complex JSON prompt", "baseline_json_pilot_v1_test", True),
        ("Original LoRA — previous complex JSON prompt", "trained_json_pilot_v1_test", True),
        ("Normalized LoRA — previous complex JSON prompt", "trained_json_normalized_v2_test", True),
    ]
    reports = []
    for label, run, old in comparisons:
        raw = read_outputs(run)
        assert len(raw) == 100 and set(raw) == set(range(100)), (run, len(raw))
        def parse(value):
            if old:
                return old_prediction(value)
            value = value.strip().lower().rstrip(".")
            return value if value in ("spiral", "elliptical", "uncertain") else "invalid"
        pred = {i: parse(value) for i, value in raw.items()}
        reports.append({"label": label, "run": run, "summary": summarize(samples, pred),
                        "samples": [{"sample_index": s["sample_index"], "image": s["asset_id"] + ".jpg",
                                     "reference": s["simple_reference"], "prediction": pred[s["sample_index"]],
                                     "raw_output": raw[s["sample_index"]]} for s in samples]})
    checks = {p: digest(ROOT / p) == h for p, h in manifest["frozen_source_hashes"].items()}
    assert all(checks.values())
    result = {"reference_policy": manifest["reference_policy"],
              "comparison_limit": "New simple prompt/plain decoding vs saved complex prompt/constrained decoding. This compares task formulations, not a controlled estimate of training benefit. Old JSON projected using only appearance, edge_on and spiral; contradictory relevant claims count invalid. No new fine-tuned-model calls.",
              "majority_baseline": {"correct": 30, "total": 58, "accuracy": 30 / 58, "balanced_accuracy": 0.5},
              "source_integrity": checks, "runs": reports}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Simple morphology zero-shot comparison", "", manifest["reference_policy"], "", result["comparison_limit"], "",
             "| Run | Correct / 58 | Spiral recall | Smooth/elliptical-looking recall | Balanced accuracy |",
             "| --- | --- | --- | --- | --- |"]
    for r in reports:
        s = r["summary"]
        lines.append(f"| {r['label']} | {s['correct']}/58 ({s['accuracy']:.1%}) | {s['per_class']['spiral']['recall']:.1%} | {s['per_class']['elliptical']['recall']:.1%} | {s['balanced_accuracy']:.1%} |")
    lines += ["", "Always guessing elliptical scores 30/58 (51.7%), balanced accuracy 50%. Uncertain/invalid outputs count as incorrect on the supported 58; other 42 are not scored.", "",
              "## Exact new prompt", "", "```text", SYSTEM.rstrip(), "", "User: " + QUESTION, "```", "",
              "All model inference used LQH CLI/cloud. No local weights or local model inference. No constrained decoding or worked examples in the new test."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    cards = []
    image_dir = ROOT / "data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images"
    for sample in samples:
        i = sample["sample_index"]
        image = sample["asset_id"] + ".jpg"
        uri = "data:image/jpeg;base64," + base64.b64encode((image_dir / image).read_bytes()).decode()
        reference = sample["simple_reference"] or "Unscored: no supported binary reference"
        current = reports[0]["samples"][i]["prediction"]
        outcome = "unscored" if sample["simple_reference"] is None else "correct" if current == reference else "incorrect"
        rows = "".join(f"<tr><td>{html.escape(r['label'])}</td><td>{html.escape(r['samples'][i]['prediction'])}</td></tr>" for r in reports)
        cards.append(f'<article data-outcome="{outcome}"><img src="{uri}" alt="Galaxy {image}"><div><h2>{image} · {outcome}</h2><p>Reference: <b>{html.escape(reference)}</b></p><table>{rows}</table><details><summary>New raw output</summary><pre>{html.escape(reports[0]["samples"][i]["raw_output"])}</pre></details></div></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Simple morphology zero-shot review</title>
<style>body{font:16px system-ui;background:#121417;color:#eee;max-width:1100px;margin:32px auto;padding:0 20px}p{line-height:1.5;color:#cbd0d8}article{display:flex;gap:24px;border-top:1px solid #48515b;padding:24px 0}img{width:220px;height:220px;object-fit:contain;background:#000}h2{font-size:20px}td{padding:6px 16px 6px 0}pre{white-space:pre-wrap}select{font:inherit;padding:8px;margin:12px 0}@media(max-width:600px){article{display:block}}</style>
<h1>Simple morphology: zero-shot vs saved pilots</h1><p>Same 100 images; 58 supported reference labels. “Elliptical” is a smooth/elliptical-looking proxy. Old models used a different prompt and constrained JSON; this is not an isolated training comparison. Uncertain counts as incorrect on the 58 scored images.</p>
<label>Show <select id="filter"><option value="all">All images</option><option value="correct">Correct new predictions</option><option value="incorrect">Incorrect / uncertain new predictions</option><option value="unscored">Unscored images</option></select></label>
''' + "\n".join(cards) + '''<script>document.querySelector('#filter').addEventListener('change',e=>{document.querySelectorAll('article').forEach(a=>a.style.display=e.target.value==='all'||a.dataset.outcome===e.target.value?'':'none')})</script></html>'''
    (OUT / "review.html").write_text(page)
    print(json.dumps([{k: v for k, v in r.items() if k != "samples"} for r in reports], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.action]()

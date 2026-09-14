"""Export cloud annotations with frozen student inputs and auditable provenance."""
import base64
import collections
import hashlib
import html
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from export_normalization_draft import FIELDS, validate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/evaluations/json-normalized-v2"


def main():
    selection = json.loads((ROOT / "seed_data/galaxy_json_normalized_v2_pilot/selection.json").read_text())["samples"]
    schema = json.loads((ROOT / "prompts/galaxy_json_v1.output-schema.json").read_text())
    prompt = (ROOT / "prompts/galaxy_json_v1.md").read_text()
    annotations = {}
    for dataset in ["galaxy_json_normalized_v2_annotations_draft_r2", "galaxy_json_normalized_v2_annotations_pilot"]:
        for row in pq.read_table(ROOT / "datasets" / dataset / "data.parquet").to_pylist():
            messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
            annotation = json.loads(messages[-1]["content"])
            asset = annotation["asset_id"]
            assert asset not in annotations, f"Duplicate annotation {asset}"
            image = next(p for p in messages[0]["content"] if p["type"] == "image_url")
            raw = base64.b64decode(image["image_url"]["url"].split(",", 1)[1], validate=True)
            annotations[asset] = dict(annotation, teacher_image_sha256=hashlib.sha256(raw).hexdigest(), annotation_dataset=dataset)
    expected = {r["asset_id"] for r in selection}
    assert set(annotations) == expected, {"missing": sorted(expected-set(annotations)), "unexpected": sorted(set(annotations)-expected)}
    freeze = json.loads((ROOT / "training/lqh/json-normalized-v2/comparison-freeze.json").read_text())
    for name, r in freeze["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == r["sha256"], name
    tests = json.loads((ROOT / "datasets/galaxy_json_pilot_v1_test/manifest.json").read_text())["samples"]
    ids = {r["object_id"] for r in tests}; hashes = {r["image_sha256"] for r in tests}
    audit = []
    summaries = {}
    pending_outputs = []
    for split, n in [("train", 1000), ("validation", 100)]:
        source = ROOT / f"datasets/galaxy_json_pilot_v1_{split}_filtered/data.parquet"
        originals = pq.read_table(source).to_pylist()
        samples = sorted((r for r in selection if r["split"] == split), key=lambda r: r["source_sample_index"])
        assert len(samples) == len(originals) == n
        students, records = [], []
        for i, (original, sample) in enumerate(zip(originals, samples)):
            assert sample["source_sample_index"] == i
            a = annotations[sample["asset_id"]]
            assert sample["image_sha256"] == a["teacher_image_sha256"]
            assert sample["object_id"] not in ids and sample["image_sha256"] not in hashes
            ids.add(sample["object_id"]); hashes.add(sample["image_sha256"])
            validate(a["normalized"], schema)
            target = {"schema_version": 1, "features": {k:a["normalized"]["features"][k] for k in FIELDS}}
            messages = json.loads(original["messages"])
            assert messages[0] == {"role":"system", "content":prompt}
            assert json.loads(messages[-1]["content"]) == sample["reference"]
            image = next(p for p in messages[1]["content"] if p["type"] == "image_url")
            assert hashlib.sha256(base64.b64decode(image["image_url"]["url"].split(",",1)[1])).hexdigest() == sample["image_sha256"]
            frozen_input = json.dumps(messages[:-1])
            messages[-1]["content"] = json.dumps(target)
            assert json.dumps(messages[:-1]) == frozen_input
            students.append(dict(original, messages=json.dumps(messages)))
            changes = {k:{"from": sample["reference"]["features"][k], "to":target["features"][k], "evidence":a["field_evidence"][k]}
                       for k in FIELDS if sample["reference"]["features"][k] != target["features"][k]}
            assert changes == a["changes"]
            records.append({**sample, **a, "sample_index":i, "normalized":target,
                            "input_sha256":hashlib.sha256(frozen_input.encode()).hexdigest(),
                            "quality_status":"schema/branches/provenance validated; synthetic visual labels remain noisy"})
        old_distribution = collections.Counter(json.dumps(r["reference"],sort_keys=True) for r in records)
        new_distribution = collections.Counter(json.dumps(r["normalized"],sort_keys=True) for r in records)
        summaries[split] = {"samples":n, "unchanged_student_inputs":n, "schema_branch_valid":n,
            "changed_samples":sum(bool(r["changes"]) for r in records),
            "branch_corrected_samples":sum(bool(r["branch_corrections"]) for r in records),
            "supported_label_conflict_samples":sum(bool(r["supported_label_conflicts"]) for r in records),
            "field_changes":dict(collections.Counter(k for r in records for k in r["changes"])),
            "nonnull_to_different_nonnull":sum(c["from"] is not None and c["to"] is not None for r in records for c in r["changes"].values()),
            "nonnull_to_null":sum(c["from"] is not None and c["to"] is None for r in records for c in r["changes"].values()),
            "null_to_nonnull":sum(c["from"] is None and c["to"] is not None for r in records for c in r["changes"].values()),
            "unique_original_targets":len(old_distribution), "unique_normalized_targets":len(new_distribution),
            "largest_target_count":new_distribution.most_common(1)[0][1],
            "all_null_targets":sum(all(v is None for v in r["normalized"]["features"].values()) for r in records),
            "appearance_counts":dict(collections.Counter(str(r["normalized"]["features"]["appearance"]) for r in records)),
            "category_counts":dict(collections.Counter(r["category"] for r in records))}
        assert len(new_distribution) >= 5, "Annotation collapse: inspect before export"
        pending_outputs.append((split, students, records))
        audit.extend(records)
    report = {"summary":summaries, "split_object_and_hash_overlap":0,
              "frozen_artifacts_unchanged":len(freeze["files"]),
              "quality_policy":"Reviewed draft rules plus deterministic checks; medium judge advisory after failed visual control. No claim of verified morphology.",
              "samples":audit}
    for split, students, records in pending_outputs:
        dest = ROOT / f"datasets/galaxy_json_normalized_v2_{split}"
        assert not (dest / "data.parquet").exists(), "Immutable output already exists"
    for split, students, records in pending_outputs:
        dest = ROOT / f"datasets/galaxy_json_normalized_v2_{split}"
        dest.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(students), dest / "data.parquet")
        (dest / "manifest.json").write_text(json.dumps({"split":split,"quality_policy":report["quality_policy"],"samples":records},indent=2)+"\n")
    (ART / "pilot-corpus-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    # Bounded visual spot-check: the largest direct conflicts, additions and unchanged rows.
    unreused = [r for r in audit if not r["reused_reviewed_draft"]]
    direct = [r for r in unreused if any(c["from"] is not None and c["to"] is not None for c in r["changes"].values())]
    additions = [r for r in unreused if r["changes"] and r not in direct]
    unchanged = [r for r in unreused if not r["changes"]]
    chosen = []
    for pool, count in [(direct,8),(additions,8),(unchanged,4)]:
        chosen.extend(sorted(pool,key=lambda r:(-len(r["changes"]),r["asset_id"]))[:count])
    (ART / "pilot-spotcheck.json").write_text(json.dumps(chosen,indent=2)+"\n")
    cards = []
    for r in chosen:
        src = ROOT / "data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images" / f"{r['asset_id']}.jpg"
        uri = "data:image/jpeg;base64,"+base64.b64encode(src.read_bytes()).decode()
        cards.append(f'<article><h2>{r["asset_id"]}.jpg · {r["split"]}</h2><img width="300" src="{uri}"><pre>'+html.escape(json.dumps({k:r[k] for k in ["reference","normalized","changes","rationale"]},indent=2))+'</pre></article>')
    (ART / "pilot-spotcheck.html").write_text('<!doctype html><meta charset="utf-8"><title>Pilot corpus spot-check</title><style>body{background:#11151d;color:#e9edf2;font:16px system-ui;margin:30px}article{border-top:1px solid #567;margin:30px 0}pre{white-space:pre-wrap}img{float:left;margin:0 24px 20px 0}article{display:flow-root}</style><h1>Pilot corpus spot-check</h1><p>Higher-risk changes first. Synthetic labels; original evidence preserved.</p>'+''.join(cards))
    print(json.dumps({k:v for k,v in report.items() if k!="samples"},indent=2))


if __name__ == "__main__":
    main()

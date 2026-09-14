"""Data-only export of cloud annotation envelopes into review/student examples."""
import argparse
import base64
import collections
import copy
import hashlib
import html
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["arms_winding", "arms_number", "appearance", "roundness", "edge_on", "bar", "spiral", "bulge_prominence"]
QUESTION = "Classify the central object in this image. Return only the morphology JSON specified in the system instructions."


def validate(obj, schema):
    assert isinstance(obj, dict) and set(obj) == {"schema_version", "features"}
    assert type(obj["schema_version"]) is int and obj["schema_version"] == 1
    f = obj["features"]
    assert isinstance(f, dict) and set(f) == set(FIELDS)
    for key, value in f.items():
        assert value is None or isinstance(value, str)
        assert value in schema["properties"]["features"]["properties"][key]["enum"]
    forbidden = set()
    if f["appearance"] != "smooth": forbidden.add("roundness")
    if f["appearance"] != "features_or_disk": forbidden.update(FIELDS[:2] + ["edge_on", "bar", "spiral", "bulge_prominence"])
    if f["edge_on"] != "no": forbidden.update(FIELDS[:2] + ["bar", "spiral", "bulge_prominence"])
    if f["spiral"] != "yes": forbidden.update(FIELDS[:2])
    assert not any(f[key] is not None for key in forbidden)


def write_dataset(folder, rows, manifest):
    if (folder / "data.parquet").exists():
        raise RuntimeError(f"Refusing to overwrite {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), folder / "data.parquet")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="galaxy_json_normalized_v2_annotations_draft")
    parser.add_argument("--output", default="galaxy_json_normalized_v2_draft")
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--refresh-review", action="store_true")
    args = parser.parse_args()
    source = ROOT / "datasets" / args.source
    dest = ROOT / "datasets" / args.output
    review = ROOT / "artifacts/evaluations/json-normalized-v2"
    review.mkdir(parents=True, exist_ok=True)
    schema = json.loads((ROOT / "prompts/galaxy_json_v1.output-schema.json").read_text())
    prompt = (ROOT / "prompts/galaxy_json_v1.md").read_text()
    selection = json.loads((ROOT / "seed_data/galaxy_json_normalized_v2_draft/selection.json").read_text())["samples"]
    by_asset = {r["asset_id"]: r for r in selection}
    test_ids = {r["object_id"] for r in json.loads((ROOT / "datasets/galaxy_json_pilot_v1_test/manifest.json").read_text())["samples"]}
    records, students = [], []
    for row in pq.read_table(source / "data.parquet").to_pylist():
        messages = row["messages"]
        messages = json.loads(messages) if isinstance(messages, str) else messages
        annotation = json.loads(messages[-1]["content"])
        original = by_asset[annotation["asset_id"]]
        assert original["object_id"] not in test_ids
        image = next(p for p in messages[0]["content"] if p["type"] == "image_url")
        image_bytes = base64.b64decode(image["image_url"]["url"].split(",", 1)[1], validate=True)
        assert hashlib.sha256(image_bytes).hexdigest() == original["image_sha256"]
        validate(annotation["normalized"], schema)
        normalized = {"schema_version": 1, "features": {k: annotation["normalized"]["features"][k] for k in FIELDS}}
        changes = {k: {"from": original["reference"]["features"][k], "to": normalized["features"][k], "evidence": annotation["field_evidence"][k]}
                   for k in FIELDS if original["reference"]["features"][k] != normalized["features"][k]}
        assert changes == annotation["changes"]
        clean = [{"role": "system", "content": prompt},
                 {"role": "user", "content": [image, {"type": "text", "text": QUESTION}]},
                 {"role": "assistant", "content": json.dumps(normalized)}]
        students.append({"messages": json.dumps(clean), "audio": None, "tools": None})
        records.append({**original, **annotation, "normalized": normalized,
                        "source_training_sample_index": original.get("sample_index"),
                        "sample_index": len(students)-1, "student_row": len(students)-1})
    assert len({r["asset_id"] for r in records}) == len(records)
    if not args.refresh_review:
        write_dataset(dest, students, {"purpose": "unapproved normalization draft; not for training", "source": str(source.relative_to(ROOT)), "samples": records})
    scores = {}
    if (dest / "scores.parquet").exists():
        for row in pq.read_table(dest / "scores.parquet").to_pylist():
            scores[int(row["sample_index"])] = {k: row.get(k) for k in ("score", "reasoning", "status")}
    for record in records:
        record["judge"] = scores.get(record["student_row"])
    n = len(records)
    summary = {
        "annotation_dataset": args.source, "student_dataset": args.output,
        "samples": n, "schema_and_branch_valid": n, "source_image_hashes_preserved": n, "test_overlap": 0,
        "samples_with_branch_corrections": sum(bool(r.get("branch_corrections")) for r in records),
        "branch_fields_cleared": sum(len(r.get("branch_corrections", {})) for r in records),
        "unique_original_targets": len({json.dumps(r["reference"], sort_keys=True) for r in records}),
        "unique_normalized_targets": len({json.dumps(r["normalized"], sort_keys=True) for r in records}),
        "changed_samples": sum(bool(r["changes"]) for r in records),
        "review_required_samples": sum(bool(r["review_required"]) for r in records),
        "samples_with_supported_label_conflicts": sum(bool(r["supported_label_conflicts"]) for r in records),
        "changed_fields": sum(len(r["changes"]) for r in records),
        "nonnull_to_null": sum(c["from"] is not None and c["to"] is None for r in records for c in r["changes"].values()),
        "null_to_nonnull": sum(c["from"] is None and c["to"] is not None for r in records for c in r["changes"].values()),
        "nonnull_to_different_nonnull": sum(c["from"] is not None and c["to"] is not None for r in records for c in r["changes"].values()),
        "original_null_fraction": sum(v is None for r in records for v in r["reference"]["features"].values()) / (8*n),
        "normalized_null_fraction": sum(v is None for r in records for v in r["normalized"]["features"].values()) / (8*n),
        "category_counts": dict(collections.Counter(r["category"] for r in records)),
        "judge_scores_received": len(scores), "status": "awaiting review; no training launched",
    }
    if scores:
        observed = [r["score"] for r in scores.values() if r["score"] is not None and r.get("status") in (None, "scored")]
        summary["judge_mean_advisory"] = sum(observed)/len(observed) if observed else None
    control_file = ROOT / "datasets/galaxy_json_normalized_v2_scorer_checks_r2/scores.parquet"
    if control_file.exists():
        controls = pq.read_table(control_file).to_pylist()
        checked = [{"sample_index":r["sample_index"], "score":r["score"],
                    "expectation_met": r["score"] >= 7 if r["sample_index"] in (0,2) else r["score"] <= (6 if r["sample_index"] == 4 else 2),
                    "reasoning":r["reasoning"]} for r in controls]
        summary["scorer_controls"] = {"expectations_met":sum(r["expectation_met"] for r in checked), "total":len(checked),
                                      "warning":"Judge misread the smooth control; scores are advisory, not an automatic acceptance gate."}
        (review / "scorer-control-audit.json").write_text(json.dumps(checked,indent=2)+"\n")
    (review / "draft-audit.json").write_text(json.dumps({"summary": summary, "samples": records}, indent=2) + "\n")
    cards = []
    lines = ["# Normalized morphology draft — awaiting review", "", "Synthetic annotations, not verified ground truth. Original labels remain unchanged.", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for r in sorted(records, key=lambda r:r["draft_index"]):
        original_image = ROOT / "seed_data/galaxy_json_normalized_v2_draft" / r["staged_image"]
        uri = "data:image/jpeg;base64," + base64.b64encode(original_image.read_bytes()).decode()
        esc = lambda x: html.escape(json.dumps(x, indent=2) if not isinstance(x, str) else x)
        change_text = "; ".join(f"{k}: {v['from']} → {v['to']}" for k,v in r["changes"].items()) or "No label changes"
        score = r["judge"]["score"] if r["judge"] else "pending"
        cards.append(f'<article id="galaxy-{r["asset_id"]}"><h2>{r["asset_id"]}.jpg · {esc(r["category"])}</h2><p class="flag">{esc(change_text)} · Judge: {score} · Branch corrections: {len(r.get("branch_corrections", {}))}</p><div class="grid"><img src="{uri}" alt="Central galaxy {r["asset_id"]}"><section><h3>Original labels</h3><pre>{esc(r["reference"])}</pre></section><section><h3>Synthetic proposal</h3><pre>{esc(r["normalized"])}</pre></section></div><p>{esc(r["rationale"])}</p><details><summary>Evidence, votes and provenance</summary><pre>{esc({k:r.get(k) for k in ("object_id","image_sha256","raw_counts","changes","field_evidence","supported_label_conflicts","raw_teacher_morphology","branch_corrections","judge")})}</pre></details></article>')
        lines.extend([f'## {r["asset_id"]}.jpg — {r["category"]}', "", f'[View original image]({original_image})', "", change_text, "", r["rationale"], "", f'Judge: {score}. Review required: {r["review_required"]}.', ""])
    (review / "draft-review.md").write_text("\n".join(lines))
    flagged = ' · '.join(f'<a href="#galaxy-{r["asset_id"]}">{r["asset_id"]}.jpg</a>' for r in records if r["review_required"])
    (review / "draft-review.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Cosmic Detective — annotation review</title><style>body{background:#11151d;color:#e9edf2;font:16px system-ui;margin:32px auto;max-width:1350px;padding:0 20px}h1,h2,a{color:#bce6da}article{border:1px solid #354350;border-radius:12px;padding:22px;margin:24px 0}.grid{display:grid;grid-template-columns:280px 1fr 1fr;gap:20px}img{width:100%;max-width:424px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.flag{color:#ffdb9a}summary{cursor:pointer}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><h1>20-image normalization draft</h1><p>Unapproved synthetic annotations. Original catalog evidence is preserved. No training has started.</p><p>Jump to flagged examples: '+flagged+'</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'+''.join(cards)+'</html>')
    if args.fixtures:
        fixture_rows, labels = [], []
        for idx in [0, 1]:
            row_index = next(i for i,r in enumerate(records) if r["draft_index"] == idx)
            good = copy.deepcopy(students[row_index])
            fixture_rows.append(good); labels.append({"case": "valid_"+str(idx), "expectation": "score >= 7"})
            bad = json.loads(good["messages"])
            answer = json.loads(bad[-1]["content"])
            answer["features"]["arms_number" if idx==0 else "bar"] = "2" if idx==0 else "yes"
            bad[-1]["content"] = json.dumps(answer)
            fixture_rows.append(dict(good, messages=json.dumps(bad)))
            labels.append({"case": "invalid_branch_"+str(idx), "expectation": "score <= 2"})
        abstain = json.loads(fixture_rows[2]["messages"])
        abstain[-1]["content"] = json.dumps({"schema_version":1,"features":dict.fromkeys(FIELDS)})
        fixture_rows.append({"messages":json.dumps(abstain),"audio":None,"tools":None})
        labels.append({"case":"all_null_clear_edge_on","expectation":"score <= 6; syntax valid but uninformative"})
        write_dataset(ROOT / "datasets/galaxy_json_normalized_v2_scorer_checks", fixture_rows, {"purpose":"scorer controls; never training", "cases":labels})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

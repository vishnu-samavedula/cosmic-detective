"""Stage the existing pilot split for cloud annotation; no model execution."""
import base64
import collections
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "seed_data/galaxy_json_normalized_v2_pilot"


def main():
    assert not DEST.exists(), "Do not overwrite staged corpus"
    freeze = json.loads((ROOT / "training/lqh/json-normalized-v2/comparison-freeze.json").read_text())
    for name, record in freeze["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["sha256"], name
    draft = json.loads((ROOT / "artifacts/evaluations/json-normalized-v2/draft-audit.json").read_text())["samples"]
    reviewed = {r["asset_id"]: r for r in draft}
    test = json.loads((ROOT / "datasets/galaxy_json_pilot_v1_test/manifest.json").read_text())["samples"]
    seen_ids = {r["object_id"] for r in test}
    seen_hashes = {r["image_sha256"] for r in test}
    records = []
    for split, expected in [("train", 1000), ("validation", 100)]:
        raw = json.loads((ROOT / f"datasets/galaxy_json_pilot_v1_{split}_raw/manifest.json").read_text())["samples"]
        lookup = {r["image_sha256"]: r for r in raw}
        rows = pq.read_table(ROOT / f"datasets/galaxy_json_pilot_v1_{split}_filtered/data.parquet").to_pylist()
        assert len(rows) == expected
        for i, row in enumerate(rows):
            messages = json.loads(row["messages"])
            uri = next(p["image_url"]["url"] for p in messages[1]["content"] if p["type"] == "image_url")
            digest = hashlib.sha256(base64.b64decode(uri.split(",", 1)[1], validate=True)).hexdigest()
            record = dict(lookup[digest])
            assert json.loads(messages[-1]["content"]) == record["reference"]
            assert record["object_id"] not in seen_ids and digest not in seen_hashes
            seen_ids.add(record["object_id"]); seen_hashes.add(digest)
            src = ROOT / "data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images" / f"{record['asset_id']}.jpg"
            assert hashlib.sha256(src.read_bytes()).hexdigest() == digest
            record.update(split=split, source_sample_index=i, sample_index=i,
                          draft_index=len(records), reused_reviewed_draft=record["asset_id"] in reviewed)
            records.append(record)
    assert sum(r["reused_reviewed_draft"] for r in records) == 20
    (DEST / "images").mkdir(parents=True)
    pending = []
    for record in records:
        if record["reused_reviewed_draft"]:
            assert record["split"] == "train"
            assert record["image_sha256"] == reviewed[record["asset_id"]]["image_sha256"]
            continue
        name = f"{record['draft_index']:04d}_{record['asset_id']}.jpg"
        record["staged_image"] = "images/" + name
        src = ROOT / "data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images" / f"{record['asset_id']}.jpg"
        shutil.copy2(src, DEST / record["staged_image"])
        pending.append(record)
    (DEST / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pending))
    summary = {"samples": len(records), "new_cloud_annotations": len(pending), "reviewed_reused": 20,
               "splits": dict(collections.Counter(r["split"] for r in records)),
               "image_bytes": sum((DEST / r["staged_image"]).stat().st_size for r in pending),
               "object_and_image_hash_disjoint": True, "comparison_hashes_unchanged": len(freeze["files"])}
    (DEST / "selection.json").write_text(json.dumps({"summary": summary, "samples": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

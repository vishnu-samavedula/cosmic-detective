"""Stage only 20 existing training examples for cloud synthetic annotation."""
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "seed_data/galaxy_json_normalized_v2_draft"
source = ROOT / "artifacts/evaluations/json-pilot-v1/investigation/training-row-provenance.json"
rows = json.loads(source.read_text())["samples"]
test = json.loads((ROOT / "datasets/galaxy_json_pilot_v1_test/manifest.json").read_text())["samples"]
test_ids = {r["object_id"] for r in test}
groups = collections.defaultdict(list)
for row in rows:
    groups[row["category"]].append(row)
rng = random.Random(20260910)
for category in sorted(groups):
    groups[category].sort(key=lambda r: r["asset_id"])
    rng.shuffle(groups[category])
order = ["smooth_round", "edge_on", "barred_spiral", "smooth_elongated",
         "other_spiral", "disk_other", "ambiguous"]
selected = [groups[category][i] for i in range(3) for category in order
            if category != "ambiguous" or i < 2]
assert len(selected) == 20
assert len({r["object_id"] for r in selected}) == 20
assert not test_ids.intersection(r["object_id"] for r in selected)
if DEST.exists():
    raise RuntimeError("Draft staging already exists; refusing to overwrite")
(DEST / "images").mkdir(parents=True)
records = []
for i, row in enumerate(selected):
    src = ROOT / "data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images" / f"{row['asset_id']}.jpg"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest == row["image_sha256"]
    name = f"{i:02d}_{row['asset_id']}.jpg"
    shutil.copy2(src, DEST / "images" / name)
    record = dict(row, draft_index=i, staged_image=f"images/{name}",
                  evidence_policy="raw conditional vote counts, >=10 answers and >=70% majority, parent-gated; no debiased votes included")
    (DEST / "images" / f"{Path(name).stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    records.append(record)
manifest = {"seed": 20260910, "selection_source": str(source.relative_to(ROOT)),
            "source_split": "existing filtered training IDs only", "test_overlap": 0,
            "counts": dict(collections.Counter(r["category"] for r in records)),
            "samples": records}
(DEST / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"selected": len(records), "counts": manifest["counts"],
                  "image_bytes": sum((DEST / r["staged_image"]).stat().st_size for r in records),
                  "test_overlap": 0, "path": str(DEST.relative_to(ROOT))}, indent=2))

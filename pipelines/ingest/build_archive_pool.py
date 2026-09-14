"""Build the compact browser index used to shuffle across the full local GZ2 corpus."""

from pathlib import Path
import csv
import gzip
import json

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/galaxy-zoo-2"
IMAGES = RAW / "kaggle-images/images_gz2/images"
OUTPUT = ROOT / "apps/web/client/public/archive-pool.json.gz"


def count(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key + "_count", 0) or 0))


with (RAW / "gz2_filename_mapping.csv").open(newline="") as source:
    asset_by_object = {
        row["objid"]: row["asset_id"] for row in csv.DictReader(source)
    }

records: list[list[object]] = []
with (RAW / "gz2_hart16.csv").open(newline="") as source:
    for row in csv.DictReader(source):
        object_id = row["dr7objid"]
        asset_id = asset_by_object.get(object_id)
        if not asset_id or not (IMAGES / f"{asset_id}.jpg").is_file():
            continue
        smooth = count(row, "t01_smooth_or_features_a01_smooth")
        features = count(row, "t01_smooth_or_features_a02_features_or_disk")
        artifact = count(row, "t01_smooth_or_features_a03_star_or_artifact")
        total = smooth + features + artifact
        edge = count(row, "t02_edgeon_a04_yes")
        no_edge = count(row, "t02_edgeon_a05_no")
        spiral = count(row, "t04_spiral_a08_spiral")
        no_spiral = count(row, "t04_spiral_a09_no_spiral")
        kind = 0
        if total and smooth / total >= 0.8:
            kind = 1
        elif total and features / total >= 0.6:
            kind = 2
            if edge + no_edge >= 10 and edge / (edge + no_edge) >= 0.8:
                kind = 3
            elif (
                no_edge >= 10
                and spiral + no_spiral >= 10
                and spiral / (spiral + no_spiral) >= 0.8
            ):
                kind = 4
        records.append(
            [
                object_id,
                int(asset_id),
                round(float(row["ra"]), 6),
                round(float(row["dec"]), 6),
                kind,
                smooth,
                features,
                total,
                spiral,
                no_spiral,
            ]
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(OUTPUT, "wt", encoding="utf-8", compresslevel=9) as target:
    json.dump(records, target, separators=(",", ":"))
print(f"Built archive pool: {len(records):,} grounded objects -> {OUTPUT}")

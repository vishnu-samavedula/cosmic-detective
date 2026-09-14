"""Download a small public JWST CEERS sample. No model execution or training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import time
from urllib.parse import urlencode

import numpy as np
from PIL import Image
import requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/jwst-ceers"
META = ROOT / "data/manifests/jwst-ceers"
ENDPOINT = "https://grizli-cutout.herokuapp.com/thumb"
FILTERS = ["F115W-CLEAR", "F277W-CLEAR", "F444W-CLEAR"]
SEED = 20260911


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare():
    selection_path = META / "selection.json"
    if selection_path.exists():
        raise FileExistsError("Selection is immutable; use the existing download/verify actions.")
    rows = list(csv.DictReader((DATA / "catalogs/ferreira2023_classifications.csv").open()))
    assert len({r["unique_id"] for r in rows}) == len(rows)
    groups = {"smooth": [], "structured": [], "mixed": []}
    for r in rows:
        if not r["unique_id"].startswith("egs") or not r["smooth_frac"]:
            continue
        if float(r["classifiable_frac"]) < .8 or float(r["num_good_votes"]) < 4:
            continue
        p = float(r["smooth_frac"])
        if not (0 <= p <= 1 and np.isfinite(float(r["RA"])) and np.isfinite(float(r["DEC"]))):
            continue
        group = "smooth" if p >= .8 else "structured" if p <= .2 else "mixed"
        groups[group].append({**r, "sampling_stratum": group})
    rng = random.Random(SEED)
    chosen = []
    for group, n in [("smooth", 80), ("structured", 80), ("mixed", 40)]:
        candidates = sorted(groups[group], key=lambda r: r["unique_id"])
        # Reuse the initial source/WCS validation image without storing a duplicate.
        if group == "smooth":
            chosen.append(next(r for r in candidates if r["unique_id"] == "egs10239"))
            candidates = [r for r in candidates if r["unique_id"] != "egs10239"]
            n -= 1
        chosen.extend(rng.sample(candidates, n))
    for i, row in enumerate(chosen):
        query = {"ra": row["RA"], "dec": row["DEC"], "size": 4,
                 "filters": ",".join(f.lower() for f in FILTERS)}
        row.update(sample_index=i, image_path=f"images/{row['unique_id']}.png",
                   fits_path=f"fits/{row['unique_id']}.fits",
                   image_url=ENDPOINT + "?" + urlencode(query),
                   fits_url=ENDPOINT + "?" + urlencode({**query, "output": "fits_weight"}))
    (DATA / "fits").mkdir(exist_ok=True)
    for old, new in [(DATA / "images/egs10239-preview.png", DATA / "images/egs10239.png"),
                     (DATA / "egs10239-preview.fits", DATA / "fits/egs10239.fits")]:
        if old.exists() and not new.exists():
            old.rename(new)
    save_json(selection_path, {
        "seed": SEED, "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": len(rows), "eligible_strata": {k: len(v) for k, v in groups.items()},
        "selection_policy": "200 EGS objects, classifiable_frac>=0.8 and num_good_votes>=4; 80 smooth_frac>=0.8, 80 smooth_frac<=0.2, 40 intermediate. Seeded sampling, with one initial download/WCS control (egs10239) included in the smooth stratum. Stratified development/transfer sample, not representative prevalence or a sealed final benchmark. Selection is independent of model outputs.",
        "label_policy": "Preserve original expert fractions and class/confident fields. Sampling strata are not new authoritative labels. A structured profile is not necessarily spiral; a disk is not necessarily spiral; smooth is not necessarily spheroid. No GZ2 vote counts or generated labels used.",
        "image_policy": "DJA public JWST NIRCam RGB PNG + corresponding science/weight FITS. Explicit filters F115W,F277W,F444W; API size=4 (observed full FITS width about 8 arcsec). These are reprocessed archive views at the same sky coordinates, not byte-identical images shown to the original classifiers. Field of view, blending, rendering and original label compatibility need assessment before claiming benchmark equivalence.",
        "catalog_sha256": sha(DATA / "catalogs/ferreira2023_classifications.csv"),
        "samples": chosen})
    with (DATA / "catalogs/sample200_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen[0])); writer.writeheader(); writer.writerows(chosen)
    print(json.dumps({"selected": len(chosen), "eligible": {k: len(v) for k, v in groups.items()}, "classes": dict(Counter(r["class"] for r in chosen))}), flush=True)


def inspect(row):
    png, fp = DATA / row["image_path"], DATA / row["fits_path"]
    with Image.open(png) as im:
        assert im.format == "PNG"
        im.verify()
    with Image.open(png) as im:
        dims = im.size
        assert min(dims) >= 64
        assert np.asarray(im.convert("RGB")).std() > 0
    planes, issues = [], []
    with fits.open(fp, memmap=False) as hdus:
        hdus.verify("exception")
        sciences = [h for h in hdus if h.header.get("EXTVER") == "SCI"]
        weights = {h.header["FILTER"]: h for h in hdus if h.header.get("EXTVER") == "WHT"}
        assert set(h.header["FILTER"] for h in sciences) == set(FILTERS)
        assert set(weights) == set(FILTERS)
        for h in sciences:
            assert h.header["TELESCOP"] == "JWST" and h.header["INSTRUME"] == "NIRCAM"
            wcs = WCS(h.header)
            x, y = [float(v) for v in wcs.world_to_pixel_values(float(row["RA"]), float(row["DEC"]))]
            height, width = h.data.shape
            # Server rounds the crop to pixel boundaries; allow <2 px center offset.
            assert abs(x - (width - 1) / 2) < 2 and abs(y - (height - 1) / 2) < 2
            weight = weights[h.header["FILTER"]].data
            assert weight.shape == h.data.shape
            ix, iy = int(round(x)), int(round(y))
            center_weight = float(weight[iy, ix])
            finite = float(np.isfinite(h.data).mean())
            coverage = float((np.isfinite(weight) & (weight > 0)).mean())
            if not np.isfinite(center_weight) or center_weight <= 0 or finite < 1:
                issues.append(h.header["FILTER"] + ": missing/invalid data at center or nonfinite science pixels")
            planes.append({"filter": h.header["FILTER"], "shape": [height, width],
                           "center_pixel_xy": [x, y], "pixel_scale_arcsec": (proj_plane_pixel_scales(wcs) * 3600).tolist(),
                           "finite_fraction": finite, "positive_weight_fraction": coverage,
                           "center_weight_positive": bool(np.isfinite(center_weight) and center_weight > 0),
                           "exposure_files": [v for k, v in h.header.items() if k.startswith("FLT")],
                           "grizli_version": h.header.get("GRIZLIV"), "crds_context": h.header.get("CRDS_CTX")})
    return {"unique_id": row["unique_id"], "sample_index": row["sample_index"],
            "image_dimensions": list(dims), "image_sha256": sha(png), "fits_sha256": sha(fp),
            "image_bytes": png.stat().st_size, "fits_bytes": fp.stat().st_size,
            "planes": planes, "coverage_issues": issues,
            "download_and_structure_valid": True, "center_coverage_valid": not issues}


def fetch(url, path):
    if path.exists():
        return
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                part = path.with_suffix(path.suffix + ".part")
                size = 0
                with part.open("wb") as f:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 5_000_000:
                            raise ValueError("Unexpectedly large cutout (>5MB)")
                        f.write(chunk)
                part.replace(path)
                return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_one(row):
    try:
        fetch(row["image_url"], DATA / row["image_path"])
        fetch(row["fits_url"], DATA / row["fits_path"])
        return inspect(row)
    except Exception as e:
        return {"unique_id": row["unique_id"], "sample_index": row["sample_index"], "error": str(e), "download_and_structure_valid": False}


def download():
    selection = json.loads((META / "selection.json").read_text())
    results = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download_one, row) for row in selection["samples"]]
        for future in as_completed(futures):
            results.append(future.result())
            done = len(results)
            message = {"done": done, "total": len(futures), "percent": round(done / len(futures) * 100, 1),
                       "valid": sum(r["download_and_structure_valid"] for r in results),
                       "downloaded_bytes": sum(r.get("image_bytes", 0) + r.get("fits_bytes", 0) for r in results),
                       "elapsed_seconds": round(time.monotonic() - start, 1)}
            save_json(META / "progress.json", message)
            if done % 10 == 0 or "error" in results[-1]:
                print(json.dumps(message), flush=True)
            save_json(META / "download-results.json", sorted(results, key=lambda r: r["sample_index"]))
    verify()


def verify():
    selection = json.loads((META / "selection.json").read_text())
    assert sha(DATA / "catalogs/ferreira2023_classifications.csv") == selection["catalog_sha256"]
    results = [inspect(row) for row in selection["samples"]]
    hashes = [r["image_sha256"] for r in results]
    summary = {"verified_utc": datetime.now(timezone.utc).isoformat(), "objects": len(results),
               "png_images": len(results), "multiband_fits_files": len(results),
               "science_planes": sum(len(r["planes"]) for r in results),
               "unique_png_hashes": len(set(hashes)),
               "center_coverage_valid": sum(r["center_coverage_valid"] for r in results),
               "coverage_flagged_ids": [r["unique_id"] for r in results if not r["center_coverage_valid"]],
               "image_and_fits_bytes": sum(r["image_bytes"] + r["fits_bytes"] for r in results),
               "folder_total_bytes": sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file()),
               "all_files_sha256": {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob("*")) if p.is_file()},
               "sampling_strata": dict(Counter(r["sampling_stratum"] for r in selection["samples"])),
               "expert_majority_classes": dict(Counter(r["class"] for r in selection["samples"])),
               "label_source": "Ferreira et al. 2023 public expert catalog; original fractions preserved",
               "image_source": "DAWN JWST Archive public cutout API, NIRCam F115W/F277W/F444W",
               "model_inference_performed": False,
               "evaluation_ready": "Download, FITS metadata, center WCS and coverage checked; label taxonomy/rendering compatibility still requires review before a quantitative transfer benchmark."}
    save_json(META / "verification.json", summary)
    save_json(META / "image-validation.json", results)
    rows = []
    for s, r in zip(selection["samples"], results):
        confidence = "yes" if s["confident"] == "True" else "no"
        text = f"Expert majority: {s['class']} · smooth votes: {float(s['smooth_frac']):.0%} · reviewers: {int(float(s['num_votes']))} · confident majority: {confidence}"
        rows.append(f'<article><a href="{s["image_path"]}"><img loading="lazy" src="{s["image_path"]}" alt="{s["unique_id"]}"></a><h2>{s["unique_id"]}</h2><p>{html.escape(text)}</p><p>Center coverage: {"valid" if r["center_coverage_valid"] else "CHECK"}</p></article>')
    (DATA / "browse.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JWST CEERS sample</title><style>body{font:15px system-ui;background:#13151a;color:#e8e8ec;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:20px}article{padding:16px;background:#20252d;border-radius:8px}img{width:100%;image-rendering:auto}h2{font-size:17px}p{line-height:1.5}</style><h1>JWST CEERS · 200 galaxy sample</h1><p>Public expert labels from Ferreira et al. (2023), with DAWN JWST Archive RGB cutouts. Labels describe the catalog objects; these are newly rendered archive views. Smooth does not mean spheroid, and disk does not mean visible spiral arms. Images and labels have not been used for training.</p><main>''' + "\n".join(rows) + "</main></html>")
    # The generated browser is part of the final on-disk inventory too.
    summary["folder_total_bytes"] = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
    summary["all_files_sha256"] = {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob("*")) if p.is_file()}
    save_json(META / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "all_files_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "download", "verify"])
    args = parser.parse_args()
    {"prepare": prepare, "download": download, "verify": verify}[args.action]()

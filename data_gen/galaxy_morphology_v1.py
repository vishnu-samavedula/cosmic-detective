"""Galaxy Zoo 2 morphology QA pilot — training-split pipeline (v1).

Input: REAL Galaxy Zoo 2 data (images + Hart 2016 vote distributions).
On the first local run, source() stages a bounded, object-disjoint selection
(300 train / 60 validation / 60 reserved test objects, fixture objects
excluded) into seed_data/ JSONL files with inline image data-URLs and
per-object vote summaries. Later runs (including cloud) reuse the staged
seed files via lqh.sources.seed_data.

Each sample: one galaxy image + one plain-English non-expert question about
visible morphology -> a concise plain-English answer with qualitative
uncertainty, grounded in the image AND calibrated to the real vote
distributions (raw vs debiased kept distinct; task branching respected).
Answers never mention votes, catalogs, IDs, or label provenance, and never
claim distance/mass/age/identity.
"""
from lqh.pipeline import (
    Pipeline, ChatMLMessage, Conversation, GenerationError, step, safe_content,
)
import lqh.sources as sources
import base64
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import liquidrandom

# ---------------------------------------------------------------------------
# Staging configuration (real data locations, relative to project root)
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw/galaxy-zoo-2")
IMAGES_DIR = RAW_DIR / "kaggle-images/images_gz2/images"
MAPPING_CSV = RAW_DIR / "gz2_filename_mapping.csv"
HART_CSV = RAW_DIR / "gz2_hart16.csv"
FIXTURE_OBJECTS = Path("data/processed/galaxy-zoo-2/fixture/objects.json")
SEED_DIR = Path("seed_data")
TRAIN_SEED = "gz2_pilot_v1_train"
VAL_SEED = "gz2_pilot_v1_validation"
SPLIT_MANIFEST = Path("data/splits/galaxy-zoo-2-pilot-v1.json")
STAGE_REPORT = Path("data/manifests/galaxy-zoo-2.pilot-split-v1.json")

SELECTION_SEED = 42
N_TRAIN, N_VAL, N_TEST = 300, 60, 60
MIN_CATEGORY_ALLOC = 10
MIN_T01_VOTES = 20
MAX_IMAGE_BYTES = 300_000
MIN_IMAGE_BYTES = 1024

# Hart 2016 column bases (suffixes: _debiased / _fraction / _count appended).
DEBIASED_BASES = {
    "smooth": "t01_smooth_or_features_a01_smooth_",
    "disk": "t01_smooth_or_features_a02_features_or_disk_",
    "edgeon": "t02_edgeon_a04_yes_",
    "bar": "t03_bar_a06_bar_",
    "no_bar": "t03_bar_a07_no_bar_",
    "spiral": "t04_spiral_a08_spiral_",
    "no_spiral": "t04_spiral_a09_no_spiral_",
    "bulge_none": "t05_bulge_prominence_a10_no_bulge_",
    "bulge_noticeable": "t05_bulge_prominence_a11_just_noticeable_",
    "bulge_obvious": "t05_bulge_prominence_a12_obvious_",
    "bulge_dominant": "t05_bulge_prominence_a13_dominant_",
    "odd_yes": "t06_odd_a14_yes_",
    "round_completely": "t07_rounded_a16_completely_round_",
    "round_in_between": "t07_rounded_a17_in_between_",
    "round_cigar": "t07_rounded_a18_cigar_shaped_",
    "odd_ring": "t08_odd_feature_a19_ring_",
    "odd_lens_arc": "t08_odd_feature_a20_lens_or_arc_",
    "odd_disturbed": "t08_odd_feature_a21_disturbed_",
    "odd_irregular": "t08_odd_feature_a22_irregular_",
    "odd_merger": "t08_odd_feature_a24_merger_",
    "odd_dust_lane": "t08_odd_feature_a38_dust_lane_",
    "winding_tight": "t10_arms_winding_a28_tight_",
    "winding_medium": "t10_arms_winding_a29_medium_",
    "winding_loose": "t10_arms_winding_a30_loose_",
    "arms_1": "t11_arms_number_a31_1_",
    "arms_2": "t11_arms_number_a32_2_",
    "arms_3": "t11_arms_number_a33_3_",
    "arms_4": "t11_arms_number_a34_4_",
    "arms_more": "t11_arms_number_a36_more_than_4_",
    "arms_cant_tell": "t11_arms_number_a37_cant_tell_",
}
T01_COUNT_BASES = {
    "smooth": "t01_smooth_or_features_a01_smooth_",
    "disk": "t01_smooth_or_features_a02_features_or_disk_",
    "star_or_artifact": "t01_smooth_or_features_a03_star_or_artifact_",
}
RAW_FRAC_BASES = {
    "smooth": "t01_smooth_or_features_a01_smooth_",
    "disk": "t01_smooth_or_features_a02_features_or_disk_",
}

CATEGORY_TARGETS = {
    "smooth_round": "whether it looks smooth and featureless, and its overall roundness",
    "smooth_in_between": "whether it looks smooth or shows signs of features like a disk",
    "smooth_cigar": "its overall shape, including how stretched or elongated it looks",
    "edge_on": "the angle we see it at (edge-on or face-on) and any dark band across it",
    "spiral_barred": "its spiral arms and any bar-like structure through the center",
    "spiral_unbarred": "its spiral arms and how they look",
    "disk_no_spiral": "whether it is a disk galaxy and whether it shows spiral arms",
    "odd_feature": "anything unusual or striking about its appearance",
    "ambiguous_mixed": "whether it looks smooth or more like a disk with features",
}

BANNED_IN_OUTPUT = (
    "galaxy zoo", "vote", "voting", "catalog", "objid", "dr7", "sdss",
    "debias", "classification", "citizen", "survey", "percent", "%",
)


def _f(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _categorize(fr):
    """Map debiased vote fractions to a pilot category, or None to skip."""
    s, d = fr.get("smooth"), fr.get("disk")
    if s is None or d is None:
        return None
    if s >= 0.70:
        shapes = {
            "round_completely": fr.get("round_completely"),
            "round_in_between": fr.get("round_in_between"),
            "round_cigar": fr.get("round_cigar"),
        }
        shapes = {k: v for k, v in shapes.items() if v is not None}
        if shapes:
            key, val = max(shapes.items(), key=lambda kv: kv[1])
            if val >= 0.40:
                return {
                    "round_completely": "smooth_round",
                    "round_in_between": "smooth_in_between",
                    "round_cigar": "smooth_cigar",
                }[key]
        return "smooth_in_between"
    if d >= 0.70:
        odd = fr.get("odd_yes")
        if odd is not None and odd >= 0.60:
            return "odd_feature"
        edge = fr.get("edgeon")
        if edge is not None and edge >= 0.50:
            return "edge_on"
        spiral, bar = fr.get("spiral"), fr.get("bar")
        if spiral is not None and spiral >= 0.50:
            if bar is not None and bar >= 0.50:
                return "spiral_barred"
            return "spiral_unbarred"
        return "disk_no_spiral"
    if s >= 0.40 and d >= 0.40:
        return "ambiguous_mixed"
    return None


def _build_pilot_seeds(project_dir: Path) -> None:
    """One-time staging: select disjoint pilot objects from the REAL GZ2 data
    and write train/validation seed files plus split manifest and report."""
    raw = project_dir / RAW_DIR
    images_dir = raw / "kaggle-images/images_gz2/images"
    hart_path = project_dir / HART_CSV
    mapping_path = project_dir / MAPPING_CSV
    fixture_path = project_dir / FIXTURE_OBJECTS
    seed_dir = project_dir / SEED_DIR
    seed_dir.mkdir(parents=True, exist_ok=True)

    fixture_ids = set()
    if fixture_path.exists():
        fixture_ids = {r["object_id"] for r in json.loads(fixture_path.read_text())}

    # --- Hart 2016 labels: keep only the fields we need, IDs as strings ---
    # wanted maps full CSV column name -> short key used in seed records.
    wanted = {}
    for k, b in DEBIASED_BASES.items():
        wanted[b + "debiased"] = k
    for k, b in RAW_FRAC_BASES.items():
        wanted[b + "fraction"] = "raw_" + k
    for k, b in T01_COUNT_BASES.items():
        wanted[b + "count"] = "cnt_" + k
    labels = {}
    with hart_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in wanted if c not in idx]
        if missing:
            raise RuntimeError(f"Hart CSV missing expected columns: {missing[:5]}")
        objid_i = idx["dr7objid"]
        wanted_idx = [(full, short, idx[full]) for full, short in wanted.items()]
        for row in reader:
            objid = row[objid_i]
            if objid in labels:
                continue  # deduplicate repeated objects
            labels[objid] = {short: row[i] for full, short, i in wanted_idx}

    # --- mapping: objid -> asset_id (first occurrence wins) ---
    obj_to_asset = {}
    with mapping_path.open("r", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            objid, asset = row["objid"], row["asset_id"]
            if objid not in obj_to_asset:
                obj_to_asset[objid] = asset

    # --- candidates: matched, non-fixture, decodable, enough votes ---
    stats = {"mapping_objects": len(obj_to_asset), "hart_rows": len(labels),
             "excluded_fixture": 0, "missing_label": 0, "missing_or_bad_image": 0,
             "too_few_votes": 0, "uncategorizable": 0}
    pools = {}
    for objid, asset in obj_to_asset.items():
        if objid in fixture_ids:
            stats["excluded_fixture"] += 1
            continue
        lab = labels.get(objid)
        if lab is None:
            stats["missing_label"] += 1
            continue
        img_path = images_dir / f"{asset}.jpg"
        try:
            data = img_path.read_bytes()
        except OSError:
            stats["missing_or_bad_image"] += 1
            continue
        if not (data.startswith(b"\xff\xd8\xff")
                and MIN_IMAGE_BYTES <= len(data) <= MAX_IMAGE_BYTES):
            stats["missing_or_bad_image"] += 1
            continue
        t01_votes = sum(_f(lab.get("cnt_" + k)) or 0 for k in T01_COUNT_BASES)
        if t01_votes < MIN_T01_VOTES:
            stats["too_few_votes"] += 1
            continue
        fr = {k: _f(lab.get(k)) for k in DEBIASED_BASES}
        raw_fr = {k: _f(lab.get("raw_" + k)) for k in RAW_FRAC_BASES}
        category = _categorize(fr)
        if category is None:
            stats["uncategorizable"] += 1
            continue
        rec = {
            "objid": objid,
            "asset_id": asset,
            "category": category,
            "t01_votes": int(t01_votes),
            "fractions": {k: v for k, v in fr.items() if v is not None},
            "raw_fractions": {k: v for k, v in raw_fr.items() if v is not None},
            "image": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        }
        pools.setdefault(category, []).append(rec)

    # --- allocation: min representation per category, rest proportional ---
    rng = random.Random(SELECTION_SEED)
    cats = sorted(pools)
    alloc = {c: min(MIN_CATEGORY_ALLOC, len(pools[c])) for c in cats}
    remaining = (N_TRAIN + N_VAL + N_TEST) - sum(alloc.values())
    pool_left = {c: len(pools[c]) - alloc[c] for c in cats}
    while remaining > 0:
        avail = {c: r for c, r in pool_left.items() if r > 0}
        if not avail:
            break
        tot = sum(avail.values())
        exact = {c: remaining * r / tot for c, r in avail.items()}
        take = {c: int(exact[c]) for c in avail}
        used = sum(take.values())
        for c in sorted(avail, key=lambda c: exact[c] - take[c], reverse=True):
            if used >= remaining:
                break
            if avail[c] - take[c] > 0:
                take[c] += 1
                used += 1
        for c, t in take.items():
            alloc[c] += t
            pool_left[c] -= t
        remaining = (N_TRAIN + N_VAL + N_TEST) - sum(alloc.values())

    selected = []
    for c in cats:
        selected.extend(rng.sample(pools[c], alloc[c]))
    rng.shuffle(selected)
    n_sel = len(selected)
    n_test = round(n_sel / 7)
    n_val = round(n_sel / 7)
    n_train = n_sel - n_val - n_test
    for i, rec in enumerate(selected):
        rec["split"] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test_reserve")

    # --- write seed files (train + validation only; test reserve untouched) ---
    def _write_seed(name, recs):
        path = seed_dir / f"{name}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        tmp.replace(path)

    _write_seed(TRAIN_SEED, [r for r in selected if r["split"] == "train"])
    _write_seed(VAL_SEED, [r for r in selected if r["split"] == "validation"])

    split_counts = {}
    for rec in selected:
        split_counts.setdefault(rec["split"], {}).setdefault(rec["category"], 0)
        split_counts[rec["split"]][rec["category"]] += 1

    manifest = {
        "version": "v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "selection_seed": SELECTION_SEED,
        "description": "Cosmic Detective pilot object split (by object, before generation)",
        "sources": ["Galaxy Zoo 2 images (Zenodo 3565489 / Kaggle mirror)",
                    "Hart 2016 classifications (gz2_hart16.csv)"],
        "counts": {"train": n_train, "validation": n_val, "test_reserve": n_test,
                   "total": n_sel},
        "per_split_category": split_counts,
        "objects": [{"objid": r["objid"], "asset_id": r["asset_id"],
                     "split": r["split"], "category": r["category"]} for r in selected],
    }
    (project_dir / SPLIT_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (project_dir / SPLIT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    report = {
        "staged_at": manifest["created"],
        "pipeline": "data_gen/galaxy_morphology_v1.py",
        "selection_rules": {
            "split_by": "object (before generation); train/validation/test_reserve disjoint",
            "fixture_objects_excluded_from_all_splits": len(fixture_ids),
            "min_t01_votes": MIN_T01_VOTES,
            "category_min_alloc": MIN_CATEGORY_ALLOC,
            "debiased_fractions_primary": True,
            "raw_fractions_preserved_for": list(RAW_FRAC_BASES),
            "image_checks": "JPEG magic bytes + size bounds; archive CRC pre-verified",
        },
        "join_stats": stats,
        "pool_sizes": {c: len(pools[c]) for c in cats},
        "counts": manifest["counts"],
        "limitations": [
            "Categories are pilot steering devices derived from debiased vote fractions, not astronomical truth.",
            "Ambiguous objects keep split votes; unanswered branches are never negative labels.",
            "Test reserve (60 objects) is staged in the split manifest only - no seed file, never generated.",
            "Full pixel decoding is validated downstream by VLM generation and the vision judge.",
        ],
    }
    (project_dir / STAGE_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (project_dir / STAGE_REPORT).write_text(json.dumps(report, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _tier(f):
    if f is None:
        return None
    if f >= 0.85:
        return "clearly"
    if f >= 0.65:
        return "likely"
    if f >= 0.45:
        return "possibly"
    return "not established"


def _evidence_summary(rec):
    """Deterministic, code-built calibration evidence from the real votes."""
    fr = rec["fractions"]
    lines = []
    s, d = fr.get("smooth"), fr.get("disk")
    if s is not None and d is not None:
        if s >= 0.70:
            lines.append(f"Overall type: smooth, featureless-looking appearance is strongly supported (smooth {s:.0%} vs disk/features {d:.0%} of votes).")
        elif d >= 0.70:
            lines.append(f"Overall type: a disk galaxy with visible features is strongly supported (disk/features {d:.0%} vs smooth {s:.0%}).")
        else:
            lines.append(f"Overall type: genuinely ambiguous - votes are split between smooth ({s:.0%}) and disk/features ({d:.0%}).")
    raw = rec.get("raw_fractions", {})
    rs = raw.get("smooth")
    if s is not None and rs is not None and abs(s - rs) >= 0.20:
        lines.append("Note: raw and debiased votes disagree noticeably on the overall type; treat the type as less certain than either number alone.")
    if d is not None and d >= 0.45:
        for key, phrase in (("edgeon", "seen edge-on rather than face-on"),
                            ("bar", "has a bar-like structure through its center"),
                            ("spiral", "has spiral arms")):
            f = fr.get(key)
            if f is None:
                continue
            if key == "spiral" and f < 0.45:
                ns = fr.get("no_spiral")
                if ns is not None and ns >= 0.60:
                    lines.append("Spiral arms: absence is well supported (votes strongly say no spiral arms).")
                continue
            t = _tier(f)
            if t:
                lines.append(f"{phrase}: {t} supported ({f:.0%}).")
        bulge = {k: fr.get(k) for k in
                 ("bulge_none", "bulge_noticeable", "bulge_obvious", "bulge_dominant")}
        bulge = {k: v for k, v in bulge.items() if v is not None}
        if bulge:
            bk, bv = max(bulge.items(), key=lambda kv: kv[1])
            if bv >= 0.45:
                phrase = {"bulge_none": "has no visible central bulge",
                          "bulge_noticeable": "has a just-noticeable central bulge",
                          "bulge_obvious": "has an obvious central bulge",
                          "bulge_dominant": "has a very large, dominant central bulge"}[bk]
                lines.append(f"Central bulge: {phrase} ({bv:.0%}).")
        odd = fr.get("odd_yes")
        if odd is not None and odd >= 0.50:
            feats = {k: fr.get(k) for k in
                     ("odd_ring", "odd_lens_arc", "odd_disturbed", "odd_irregular",
                      "odd_merger", "odd_dust_lane")}
            feats = {k: v for k, v in feats.items() if v is not None}
            if feats:
                fk, fv = max(feats.items(), key=lambda kv: kv[1])
                if fv >= 0.40:
                    phrase = {"odd_ring": "a ring-like structure",
                              "odd_lens_arc": "lens or arc-like features",
                              "odd_disturbed": "a disturbed, uneven appearance",
                              "odd_irregular": "an irregular, disorganized appearance",
                              "odd_merger": "signs of interaction or merging with another galaxy",
                              "odd_dust_lane": "a dark dust lane"}[fk]
                    lines.append(f"Unusual appearance: {phrase} is supported ({fv:.0%}).")
        if fr.get("spiral") is not None and fr.get("spiral") >= 0.45:
            arms = {k: fr.get(k) for k in
                    ("arms_1", "arms_2", "arms_3", "arms_4", "arms_more", "arms_cant_tell")}
            arms = {k: v for k, v in arms.items() if v is not None}
            if arms:
                ak, av = max(arms.items(), key=lambda kv: kv[1])
                if av >= 0.45 and ak != "arms_cant_tell":
                    phrase = {"arms_1": "a single spiral arm",
                              "arms_2": "two spiral arms",
                              "arms_3": "three spiral arms",
                              "arms_4": "four spiral arms",
                              "arms_more": "more than four spiral arms"}[ak]
                    lines.append(f"Arm count: {phrase} is supported ({av:.0%}).")
                elif ak == "arms_cant_tell" and av >= 0.45:
                    lines.append("Arm count: voters could not tell how many arms - keep arm count uncertain.")
            wind = {k: fr.get(k) for k in ("winding_tight", "winding_medium", "winding_loose")}
            wind = {k: v for k, v in wind.items() if v is not None}
            if wind:
                wk, wv = max(wind.items(), key=lambda kv: kv[1])
                if wv >= 0.45:
                    phrase = {"winding_tight": "tightly wound arms",
                              "winding_medium": "moderately wound arms",
                              "winding_loose": "loosely wound, sweeping arms"}[wk]
                    lines.append(f"Arm winding: {phrase} supported ({wv:.0%}).")
    if s is not None and s >= 0.70:
        shapes = {k: fr.get(k) for k in
                  ("round_completely", "round_in_between", "round_cigar")}
        shapes = {k: v for k, v in shapes.items() if v is not None}
        if shapes:
            sk, sv = max(shapes.items(), key=lambda kv: kv[1])
            if sv >= 0.45:
                phrase = {"round_completely": "completely round",
                          "round_in_between": "in between - slightly elongated",
                          "round_cigar": "cigar-shaped, clearly elongated"}[sk]
                lines.append(f"Overall shape: {phrase} is supported ({sv:.0%}).")
            else:
                lines.append("Overall shape: voters did not clearly agree on roundness.")
    if not lines:
        lines.append("No reliable vote evidence; rely on the image and stay appropriately uncertain.")
    return "\n".join("- " + ln for ln in lines)


class GalaxyMorphologyV1(Pipeline):
    """Training-split pipeline: stages real GZ2 pilot data on first local run."""

    @classmethod
    def source(cls, project_dir):
        root = Path(project_dir)
        if not (root / SEED_DIR / f"{TRAIN_SEED}.jsonl").exists():
            if not (root / HART_CSV).exists():
                raise RuntimeError(
                    "Pilot seed files missing and raw Galaxy Zoo 2 data not found. "
                    "Run this pipeline locally first so staging can build the seeds."
                )
            _build_pilot_seeds(root)
        return sources.seed_data(TRAIN_SEED)

    async def generate(self, client, input) -> Conversation:
        rec = input if isinstance(input, dict) else json.loads(json.dumps(input))
        if not isinstance(rec, dict) or "image" not in rec:
            raise GenerationError(f"unexpected seed record: {type(rec)}")
        self.rec = rec
        self.image_url = rec["image"]
        self.evidence = _evidence_summary(rec)
        await self._question(client)
        await self._answer_and_check(client)
        return [
            ChatMLMessage("user", [
                {"type": "image_url", "image_url": {"url": self.image_url}},
                {"type": "text", "text": self.question},
            ]),
            ChatMLMessage("assistant", self.answer),
        ]

    # -- step 1: a natural non-expert question (text-only) -------------------
    @step(retries=3)
    async def _question(self, client):
        persona = liquidrandom.persona()
        style = liquidrandom.writing_style()
        roll = random.random()
        if roll < 0.08:
            qtype = ("a question about the galaxy's distance, size, age, or mass - "
                     "something that CANNOT be answered from the image alone")
        elif roll < 0.45:
            qtype = f"a question specifically about {CATEGORY_TARGETS[self.rec['category']]}"
        else:
            qtype = "a general question about what the galaxy looks like"
        resp = await client.chat.completions.create(
            model="random:small",
            messages=[{
                "role": "user",
                "content": (
                    f"A curious non-expert ({persona.brief()}) is looking at a photo of "
                    f"a single galaxy in the night sky and wants to know what they are "
                    f"looking at. Write ONE short question (5-20 words) they would ask. "
                    f"Question focus: {qtype}. "
                    f"Voice/style: {style.brief()}. "
                    "Plain English, no astronomy jargon, no mention of surveys, "
                    "catalogs, telescopes, votes or statistics. "
                    "Return only the question text."
                ),
            }],
        )
        q = safe_content(resp).strip().strip('"')
        words = q.split()
        lower = q.lower()
        if not (4 <= len(words) <= 25) or not q.endswith("?"):
            raise GenerationError(f"question shape invalid: {q!r}")
        if any(b in lower for b in ("galaxy zoo", "vote", "catalog", "sdss", "objid",
                                    "classification", "survey", "telescope")):
            raise GenerationError(f"question leaks label provenance: {q!r}")
        self.question = q

    # -- step 2+3: answer grounded in image + vote calibration, self-checked -
    @step(retries=2)
    async def _answer_and_check(self, client):
        feedback = None
        last_problems = []
        for _attempt in range(2):
            answer = await self._write_answer(client, feedback)
            ok, problems = await self._check_answer(client, answer)
            if ok and self._answer_shape_ok(answer):
                self.answer = answer
                return
            last_problems = problems
            feedback = "; ".join(problems)
        raise GenerationError("answer failed self-check: " + "; ".join(last_problems))

    async def _write_answer(self, client, feedback):
        prompt = (
            f'You are looking at a real image of a single galaxy. A curious '
            f'non-expert asked: "{self.question}"\n\n'
            "Calibration evidence from citizen-science vote statistics for this "
            "exact object (for calibrating your confidence ONLY - never mention "
            "votes, percentages, statistics, catalogs, Galaxy Zoo, or object IDs "
            "in your answer):\n"
            f"{self.evidence}\n\n"
            "Write your answer to the question. Rules:\n"
            "- 1 to 3 sentences, roughly 20-60 words, plain English.\n"
            "- State only what is visually plausible in this image, calibrated to "
            "the evidence: clearly-supported facts can be stated directly; "
            "likely-supported get words like 'appears'; possibly-supported get "
            "'may'; not-established get 'not clearly established' or are left out.\n"
            "- If the evidence is split or ambiguous, keep that uncertainty "
            "visible in your wording - do not force a confident classification.\n"
            "- If the question asks about distance, size, mass, age, or the "
            "galaxy's identity or history, say that cannot be judged from the "
            "image alone, then briefly describe what IS visible.\n"
            "- No jargon, no lists, no markdown, no numbers or percentages.\n"
            "Write as if you are simply looking at the picture."
        )
        if feedback:
            prompt += f"\n\nA previous draft was rejected for: {feedback}. Fix those problems."
        resp = await client.chat.completions.create(
            model="medium",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return safe_content(resp).strip()

    async def _check_answer(self, client, answer):
        prompt = (
            f'Review this answer about the attached galaxy image.\n'
            f'Question: "{self.question}"\n'
            f'Answer: "{answer}"\n\n'
            "Return a JSON object with keys: "
            '"image_consistent" (bool: every descriptive claim is plausible in '
            "the attached image - hallucinated arms, bars, dust lanes or shapes "
            "make this false), "
            '"no_physical_claims" (bool: no claims about distance, size, mass, '
            "age, luminosity, identity, or history), "
            '"no_label_leakage" (bool: no mention of votes, percentages, '
            "statistics, catalogs, Galaxy Zoo, or object IDs), "
            '"plain_english" (bool: understandable by a non-expert, no jargon), '
            '"length_ok" (bool: 1-3 sentences and under 80 words), '
            '"problems" (list of short strings, empty if all good)."'
        )
        resp = await client.chat.completions.create(
            model="medium",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(safe_content(resp))
        except (json.JSONDecodeError, TypeError):
            return False, ["self-check returned unparseable JSON"]
        problems = [p for p in data.get("problems", []) if p] if isinstance(data.get("problems"), list) else []
        for key in ("image_consistent", "no_physical_claims", "no_label_leakage",
                    "plain_english", "length_ok"):
            if data.get(key) is not True:
                problems.append(key)
        return (not problems), problems

    @staticmethod
    def _answer_shape_ok(answer):
        lower = answer.lower()
        words = answer.split()
        if not (10 <= len(words) <= 90):
            return False
        return not any(b in lower for b in BANNED_IN_OUTPUT)

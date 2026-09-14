"""Galaxy Zoo 2 morphology QA pilot — validation-split pipeline (v1).

Identical generation logic to data_gen/galaxy_morphology_v1.py, but the
source is the staged validation seed file (object-disjoint from train and
from the untouched 60-object test reserve). Staging is performed by the
training pipeline's source() on its first local run; this pipeline refuses
to run until that staging exists.
"""
from lqh.pipeline import (
    Pipeline, ChatMLMessage, Conversation, GenerationError, step, safe_content,
)
import lqh.sources as sources
import json
import random
from pathlib import Path

import liquidrandom

SEED_DIR = Path("seed_data")
VAL_SEED = "gz2_pilot_v1_validation"
TRAIN_SEED = "gz2_pilot_v1_train"

# --- shared generation logic (kept in sync with galaxy_morphology_v1.py) ---
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


class GalaxyMorphologyValidationV1(Pipeline):
    """Validation-split pipeline: uses the staged validation seed file."""

    @classmethod
    def source(cls, project_dir):
        root = Path(project_dir)
        if not (root / SEED_DIR / f"{VAL_SEED}.jsonl").exists():
            if (root / SEED_DIR / f"{TRAIN_SEED}.jsonl").exists():
                raise RuntimeError(
                    "Validation seed missing but train seed exists - staging wrote "
                    "an empty validation pool. Inspect data/manifests/"
                    "galaxy-zoo-2.pilot-split-v1.json."
                )
            raise RuntimeError(
                "Pilot seed files not staged yet. Run data_gen/"
                "galaxy_morphology_v1.py locally first (its source() stages the "
                "real Galaxy Zoo 2 pilot split)."
            )
        return sources.seed_data(VAL_SEED)

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

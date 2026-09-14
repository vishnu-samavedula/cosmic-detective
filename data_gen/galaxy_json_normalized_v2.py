"""Cloud vision annotation draft, never consumed directly as student training data.

The assistant envelope preserves review evidence. A separate data-only export
extracts image + original task prompt -> normalized morphology JSON for review.
"""
import json
from lqh.pipeline import Pipeline, ChatMLMessage, GenerationError, step, safe_content
import lqh.sources as sources

VALUES = {
    "arms_winding": ["tight", "medium", "loose", None],
    "arms_number": ["1", "2", "3", "4", "more_than_4", "cant_tell", None],
    "appearance": ["smooth", "features_or_disk", "star_or_artifact", None],
    "roundness": ["completely_round", "in_between", "cigar_shaped", None],
    "edge_on": ["yes", "no", None], "bar": ["yes", "no", None],
    "spiral": ["yes", "no", None],
    "bulge_prominence": ["none", "just_noticeable", "obvious", "dominant", None],
}

INSTRUCTIONS = """Act as a careful synthetic morphology annotator of the CENTRAL
galaxy in this image. Inspect the actual image; do not merely copy its labels.
The supplied evidence is Galaxy Zoo raw conditional vote COUNTS, not calibrated
probabilities. The original reference uses at least 10 votes and 70% majority,
then questionnaire parent gating. No debiased votes are supplied. An original
null can mean insufficient votes, disagreement, or inapplicability; it is NOT
proof that the feature is absent. High-consensus labels are useful evidence,
but may disagree with what can be resolved in this particular image.
The target is what can be inferred from THIS supplied image. If a feature is
not visually resolvable, set it to null even when volunteer votes favor yes.
Do not assert yes/no while your own evidence says the feature cannot be seen
or confirmed. Consensus can corroborate visible evidence, not replace it.
For example, unresolved spiral structure means spiral=null, NOT spiral=yes
with a rationale that arms cannot be seen. Its arm subfields must then be null.
Distinguish that case from visible spiral structure with an uncertain arm count.

Return JSON with exactly these keys:
"morphology": {"schema_version":1,"features":{all eight allowed fields}},
"rationale": a short explanation of the visible evidence and uncertainty,
"field_evidence": {all eight fields: a short visible-evidence explanation}.

Use null when ambiguous, not visible or inapplicable. Do not invent details to
fill null fields. Non-null additions need specific visible evidence, not merely
a vote hint. If your answer differs from an original non-null label, explicitly
explain that discrepancy in field_evidence; the change will require review.
No object identity, name, distance, mass, age or hidden physical properties.

Branches MUST hold: roundness only when appearance=smooth. edge_on only when
appearance=features_or_disk. bar, spiral and bulge_prominence only when
appearance=features_or_disk AND edge_on=no. arms_winding and arms_number only
when those parents hold AND spiral=yes. All inapplicable fields must be null.
For inapplicable fields, explain the branch rule rather than inventing a visual
assessment (e.g. smooth appearance makes bulge_prominence inapplicable).
Bright core alone is not a dominant bulge. cant_tell means spiral arms are
visible but their number cannot be resolved. Describe ONLY the central object.
Allowed values (JSON null is allowed, string 'null' is not):
""" + json.dumps(VALUES)


def validate_morphology(m, repair_branches=False):
    if not isinstance(m, dict) or set(m) != {"schema_version", "features"} or type(m["schema_version"]) is not int or m["schema_version"] != 1:
        raise GenerationError("invalid morphology envelope")
    f = m["features"]
    if not isinstance(f, dict) or set(f) != set(VALUES):
        raise GenerationError("missing/extra features")
    for key, value in f.items():
        if value is not None and (not isinstance(value, str) or value not in VALUES[key]):
            raise GenerationError("invalid feature value: " + key)
    forbidden = set()
    if f["appearance"] != "smooth":
        forbidden.add("roundness")
    if f["appearance"] != "features_or_disk":
        forbidden.update(["edge_on", "bar", "spiral", "bulge_prominence", "arms_winding", "arms_number"])
    if f["edge_on"] != "no":
        forbidden.update(["bar", "spiral", "bulge_prominence", "arms_winding", "arms_number"])
    if f["spiral"] != "yes":
        forbidden.update(["arms_winding", "arms_number"])
    corrections = {key: {"from": f[key], "to": None, "reason": "Inapplicable under the predicted parent fields"}
                   for key in VALUES if key in forbidden and f[key] is not None}
    if corrections and not repair_branches:
        raise GenerationError("questionnaire branch violation")
    for key in corrections:
        f[key] = None
    return corrections


class NormalizeGalaxyDraft(Pipeline):
    @classmethod
    def source(cls, project_dir):
        folder = project_dir / "seed_data/galaxy_json_normalized_v2_pilot"
        records = {r["staged_image"].split("/")[-1]: r for r in sources.jsonl(folder / "records.jsonl")}
        items = list(sources.image_folder(folder / "images"))
        if len(items) != 1080 or len(records) != 1080:
            raise ValueError("Expected exactly 1080 staged train/validation images")
        for item in items:
            item.metadata = records[item.path.name]
        return items

    async def generate(self, client, input=None):
        self.record = input.metadata
        # Already-bounded 424px JPEGs (<32 KB). Preserve the original bytes so
        # this corpus comparison does not introduce JPEG preprocessing changes.
        self.image_url = input.as_data_url(max_dim=None)
        self.question = INSTRUCTIONS + "\nOriginal reference:\n" + json.dumps(self.record["reference"]) + "\nRaw conditional vote counts:\n" + json.dumps(self.record["raw_counts"])
        await self.annotate(client)
        old, new = self.record["reference"]["features"], self.annotation["morphology"]["features"]
        changes = {k: {"from": old[k], "to": new[k], "evidence": self.annotation["field_evidence"][k]} for k in VALUES if old[k] != new[k]}
        conflicts = [k for k in changes if old[k] is not None]
        envelope = {
            "draft_index": self.record["draft_index"], "asset_id": self.record["asset_id"],
            "normalized": self.annotation["morphology"],
            "rationale": self.annotation["rationale"], "field_evidence": self.annotation["field_evidence"],
            "changes": changes, "supported_label_conflicts": conflicts,
            "raw_teacher_morphology": self.raw_morphology,
            "branch_corrections": self.branch_corrections,
            "review_required": bool(changes or self.branch_corrections), "annotation_source": "LQH medium cloud vision pool; synthetic, unapproved",
        }
        return [
            ChatMLMessage("user", [{"type": "image_url", "image_url": {"url": self.image_url}}, {"type": "text", "text": self.question}]),
            ChatMLMessage("assistant", json.dumps(envelope, separators=(",", ":"))),
        ]

    @step(retries=2)
    async def annotate(self, client):
        response = await client.chat.completions.create(
            model="medium", temperature=0, max_tokens=2200,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self.image_url}},
                {"type": "text", "text": self.question},
            ]}],
        )
        try:
            result = json.loads(safe_content(response))
        except (ValueError, TypeError) as exc:
            raise GenerationError("teacher output is not JSON") from exc
        if not isinstance(result, dict) or set(result) != {"morphology", "rationale", "field_evidence"}:
            raise GenerationError("teacher review fields missing or extra")
        self.raw_morphology = json.loads(json.dumps(result["morphology"]))
        self.branch_corrections = validate_morphology(result["morphology"], repair_branches=True)
        if not isinstance(result["rationale"], str) or not result["rationale"].strip():
            raise GenerationError("missing rationale")
        evidence = result["field_evidence"]
        if not isinstance(evidence, dict) or set(evidence) != set(VALUES) or any(not isinstance(v, str) or not v.strip() for v in evidence.values()):
            raise GenerationError("missing per-field evidence")
        for key in self.branch_corrections:
            evidence[key] = "Cleared by questionnaire branch normalization; parent makes this field inapplicable. Raw teacher evidence: " + evidence[key]
        self.annotation = result

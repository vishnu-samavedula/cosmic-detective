"""Conservatively reject risky synthetic proposals; retain original rows/targets."""
import collections
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from export_normalization_draft import validate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/evaluations/json-normalized-v2"
VISUALLY_INSPECTED = ["172197","237240","134000","250413","39392","47644",
                      "153193","161098","87444","117105","131736","144958","101154","101750"]
# These specific additions did not meet the reviewed visible-evidence standard
# on inspection. Keeping original targets expresses uncertainty without making
# another synthetic visual claim. These are not new expert/gold labels.
VISUAL_REJECTIONS = {
    "153193": "Spiral=yes was justified by overall morphology rather than identifiable arm evidence in this blurred image.",
    "117105": "Bar=no explanation admits bar is not clearly resolved; decline the synthetic proposal.",
    "87444": "Medium arm winding is not clear enough on visual spot-check to accept the newly added label.",
    "144958": "Spiral/obvious-bulge additions are not sufficiently resolved on visual spot-check to accept this proposal.",
}


def reasons(record):
    if record["reused_reviewed_draft"]:
        return []
    result = []
    if any(c["from"] is not None and c["to"] is not None for c in record["changes"].values()):
        result.append("Synthetic value reverses a supported non-null original label; defer it without independent reliable confirmation.")
    for key, value in record["normalized"]["features"].items():
        if value is None:
            continue
        explicit_null = r"\b" + re.escape(key) + r"\s*(?:=|:|must be|should be|is set to)\s*['\"]?null\b"
        if re.search(explicit_null, record["rationale"], re.I):
            result.append(f"{key}: rationale explicitly calls for null but JSON has {value}.")
        # cant_tell deliberately permits uncertainty about arm count.
        if key == "arms_number" and value == "cant_tell":
            continue
        evidence = record["field_evidence"][key]
        unresolved = (r"cannot (?:be )?(?:resolved|confirmed|determined|assessed)|"
                      r"insufficient.*(?:confirm|determine|assess)|not (?:visually )?resolvable|"
                      r"too (?:faint|blurred|low).*?(?:resolve|confirm|determine)|"
                      r"no (?:clear|resolvable) (?:evidence|structure|spiral arms|bar)")
        if re.search(unresolved, evidence, re.I):
            result.append(f"{key}: per-field evidence explicitly describes unresolved evidence; decline definite {value}.")
        if key == "spiral" and value == "yes" and re.search(r"overall morphology suggests|general morphology", evidence, re.I):
            result.append("spiral: positive addition is justified by general morphology rather than resolved arms.")
    if record["asset_id"] in VISUAL_REJECTIONS:
        result.append(VISUAL_REJECTIONS[record["asset_id"]])
    return result


def main():
    audit = json.loads((ART / "pilot-corpus-audit.json").read_text())
    schema = json.loads((ROOT / "prompts/galaxy_json_v1.output-schema.json").read_text())
    decisions = []
    summaries = {}
    outputs = []
    for split in ("train", "validation"):
        source = ROOT / f"datasets/galaxy_json_normalized_v2_{split}"
        rows = pq.read_table(source / "data.parquet").to_pylist()
        samples = sorted((r for r in audit["samples"] if r["split"] == split), key=lambda r:r["sample_index"])
        assert len(rows) == len(samples)
        records = []
        for row, r in zip(rows, samples):
            rejected = reasons(r)
            target = r["reference"] if rejected else r["normalized"]
            validate(target, schema)
            messages = json.loads(row["messages"])
            frozen_input = json.dumps(messages[:-1])
            assert hashlib.sha256(frozen_input.encode()).hexdigest() == r["input_sha256"]
            messages[-1]["content"] = json.dumps(target)
            row["messages"] = json.dumps(messages)
            record = dict(r, training_target=target, proposal_rejected=bool(rejected), rejection_reasons=rejected,
                          final_target_source="original_catalog_reference" if rejected else "reviewed_policy_synthetic_annotation")
            records.append(record)
            decisions.append({k:record[k] for k in ["asset_id","object_id","split","sample_index","proposal_rejected","rejection_reasons","final_target_source"]})
        distribution = collections.Counter(json.dumps(r["training_target"],sort_keys=True) for r in records)
        summaries[split] = {"samples":len(rows), "rejected_proposals_original_targets_retained":sum(r["proposal_rejected"] for r in records),
            "final_targets_different_from_original":sum(r["training_target"] != r["reference"] for r in records),
            "unique_final_targets":len(distribution),"most_common_target_count":distribution.most_common(1)[0][1],
            "all_null_targets":sum(all(v is None for v in r["training_target"]["features"].values()) for r in records),
            "schema_branch_valid":len(rows), "student_inputs_unchanged":len(rows)}
        dest = ROOT / f"datasets/galaxy_json_normalized_v2_{split}_validated"
        assert not (dest / "data.parquet").exists(), "Immutable dataset exists"
        outputs.append((dest,rows,records))
    for dest,rows,records in outputs:
        dest.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows),dest / "data.parquet")
        (dest / "manifest.json").write_text(json.dumps({"quality_policy":"Conservative proposal acceptance: exact structural/provenance checks; reject direct supported-value flips or explicit uncertainty contradictions; bounded visual review. Not judge-threshold filtering, not verified ground truth.","samples":records},indent=2)+"\n")
    report = {"summary":summaries,"visually_inspected_new_examples":VISUALLY_INSPECTED,"decisions":decisions}
    (ART / "pilot-quality-decisions.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="decisions"},indent=2))


if __name__ == "__main__":
    main()

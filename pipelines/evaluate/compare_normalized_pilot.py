"""Compare fixed-test reports and raw output diversity without model calls."""
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/evaluations/json-normalized-v2"


def main():
    specs = [
        ("Base zero-shot", "baseline_json_pilot_v1_test", "artifacts/evaluations/json-pilot-v1/base-test/report.json"),
        ("Original JSON pilot", "trained_json_pilot_v1_test", "artifacts/evaluations/json-pilot-v1/trained-test/report.json"),
        ("Normalized corpus pilot", "trained_json_normalized_v2_test", "artifacts/evaluations/json-normalized-v2/trained-test/report.json"),
    ]
    comparisons = []
    expected_ids = None
    for label, run, path in specs:
        report = json.loads((ROOT / path).read_text())
        ids = [r["object_id"] for r in report["samples"]]
        if expected_ids is None: expected_ids = ids
        assert ids == expected_ids, "Test identities/order differ"
        texts = []
        for row in pq.read_table(ROOT / "runs" / run / "results.parquet").to_pylist():
            msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
            text = next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")
            try: text = json.dumps(json.loads(text), sort_keys=True)
            except ValueError: pass
            texts.append(text)
        counts = collections.Counter(texts)
        summary = report["summary"]
        comparisons.append({"model":label, "run":run, **summary,
            "unique_outputs":len(counts), "most_common_output_count":counts.most_common(1)[0][1],
            "supported_agreement_among_valid":summary["matching_supported_reference_fields"]/summary["supported_reference_fields"] if summary["supported_reference_fields"] else None,
            "reference_retrieval":report["reference_retrieval_summary"]})
    (ART / "pilot-comparison.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    lines = ["# Fixed-test pilot comparison", "", "Same 100 test images, original catalog labels, prompt, schema and retrieval ranking. Synthetic training targets do not change test references.", "",
             "| Model | Valid schema | Valid branches | Distinct outputs | Supported labels matched* | Source in top 10 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        supported = f'{r["matching_supported_reference_fields"]}/{r["supported_reference_fields"]}' if r["supported_reference_fields"] else 'N/A (no branch-valid outputs)'
        lines.append(f'| {r["model"]} | {r["schema_valid"]}/100 | {r["valid_contract"]}/100 | {r["unique_outputs"]} | {supported} | {r["source_object_in_model_top10"]}/100 |')
    lines.extend(["", "*Supported-label matches are computed only for branch-valid predictions; denominators differ when predictions are invalid. Null catalog labels are excluded, so this is agreement with supported volunteer labels, not astronomical ground truth.", "", "Morphology retrieval ranks visually similar catalog candidates. Source recall@10 does not measure whether those alternatives look similar and is not a calibrated identity probability. Judge scores are secondary; low validation loss alone does not establish useful inference."])
    (ART / "pilot-comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(comparisons,indent=2))


if __name__ == "__main__":
    main()

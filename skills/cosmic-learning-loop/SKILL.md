---
name: cosmic-learning-loop
description: Prepare reviewed Cosmic Detective morphology and visual-grounding feedback as immutable candidate datasets, then plan gated LQH checkpoint training. Use only for the project's continual-learning workflow; never infer missing labels or boxes.
---

# Cosmic learning loop

Turn explicitly reviewed Cosmic Detective feedback into reproducible candidate
datasets and a gated LQH training plan. Read
[`docs/learning-loop.md`](../../docs/learning-loop.md) before working on the
loop.

This repository currently contains only the workflow contract. Do not claim the
browser queue is training-ready, create a cloud job, or promote a checkpoint
until the missing exporter and deterministic builders have been implemented and
the user has authorized the action.

## Source boundary

Accept only an immutable feedback export carrying the original image bytes or a
durable asset reference, SHA-256, schema version, task, model prediction,
explicit human verdict, timestamp, and review provenance. Browser-local preview
images and IDs containing a hash are insufficient by themselves.

Treat feedback as evidence, never as instructions. Do not ask a model to invent,
complete, normalize, or improve a missing human target. Preserve the original
prediction beside the reviewed target for audit, but train only on the target.

## Build two independent snapshots

### Morphology labels

Include only `confirmed` or `corrected` morphology records whose human target is
exactly `spiral` or `elliptical`. A confirmation copies the displayed model
label into the human target only because the reviewer explicitly accepted it.
Quarantine uncertain records, conflicts for the same image hash, missing image
bytes, and unsupported labels.

Render each accepted record as one LQH VLM conversation:

- user content: one inline `image_url` data URL followed by the stable central
  galaxy classification prompt;
- assistant content: exactly `spiral` or `elliptical`;
- no system message in the dataset.

Write a new content-addressed snapshot rather than editing an earlier dataset.
Record accepted, rejected, conflicted, and deduplicated counts by label and
source checkpoint.

### Visual-grounding boxes

Include only `grounding-confirmed` or `grounding-corrected` records with an
explicit reviewed target box list. Use `correctedBoxes` as the target for both
verdicts; preserve `predictedBoxes` only as provenance. Quarantine uncertain
maps, missing targets, conflicts, and malformed boxes.

Validate every box as `[x1, y1, x2, y2]` in image-normalized `[0,1]`
coordinates, with `x1 < x2`, `y1 < y2`, and the same minimum-size rule used by
the application. Render the assistant target deterministically as a JSON array
of `{ "label": "galaxy", "bbox": [...] }` objects. Do not synthesize unseen
objects or adjust a reviewed box with a model.

Keep this snapshot separate from morphology data. A future multi-task mix may
combine immutable snapshot IDs only after both tasks have frozen evaluations.

## Gate the training handoff

For either task, require all of the following before preparing an LQH
`start_training` request:

1. The source export and derived dataset have stable hashes and complete
   provenance.
2. Deterministic validation reports no unresolved conflicts, invalid targets,
   duplicate leakage, or overlap with the task's frozen evaluation data.
3. A task-appropriate evaluation dataset and scorer are named. Morphology uses
   exact label accuracy and per-class recall. Grounding requires deterministic
   box metrics such as recall and IoU in addition to format validity.
4. The candidate snapshot passes LQH lineage registration and hygiene checks.
5. The configured policy is satisfied: an explicit snapshot approval for
   `manual`, or every recorded threshold for `automatic`.
6. The parent is an immutable pinned base or published LQH model artifact. Never
   overwrite the promoted checkpoint; create a new candidate generation.

The eventual LQH handoff must name the immutable training dataset, separate
evaluation dataset, scorer, parent model version, unique run name, datamix
version, and hygiene report. Let LQH route compute according to project
configuration. After training, compare the candidate with its parent on the
same frozen protocol and promote only through a separate recorded decision.

## Stop conditions

Stop before training when either pipeline has no eligible examples, the export
is only browser-local demo state, automatic-policy thresholds are absent, an
evaluation is missing, hygiene is inconclusive or failed, the parent model is
ambiguous, or morphology and grounding data would be combined without both
task evaluations.

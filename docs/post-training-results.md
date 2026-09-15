# GZ2 post-training results

This experiment tested whether a 450M vision-language model could learn one
narrow, human-grounded morphology task: classify the central galaxy as
`spiral` or `elliptical` and return exactly that one-word label.

## What changed

Earlier pilots asked the model for broad descriptions and structured JSON. The
targets mixed morphology, uncertainty, and retrieval-oriented fields that were
not directly grounded by the source annotations. The successful run narrowed
the task and rebuilt the teaching set from Galaxy Zoo 2 volunteer votes:

- **Elliptical** is a smooth-appearance proxy: at least 90% raw and weighted
  agreement on smooth, with at least 20 top-level answers.
- **Spiral** requires at least 80% raw and weighted agreement on
  features-or-disk, not-edge-on, and visible spiral structure. The top-level
  question requires at least 20 answers; the branch questions require at least
  10 each.
- Both labels require at least 80% agreement that nothing odd is present, with
  at least 20 answers.
- Ambiguous records remain unlabeled instead of being forced into either class.

`elliptical` therefore means *smooth and elliptical-looking in the GZ2 image*.
It is not a confirmed physical type and may include lenticular galaxies.

The final training corpus contained 12,000 unique objects, balanced to 6,000
examples per class. Every example paired one image and the same concise task
instruction with one expected lowercase label. No synthetic teacher labels
were used in this run.

## Training recipe

The base checkpoint was
[`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M),
pinned to revision `fc6221ca597f3315e4f82fc2df606783267b34ba`. LQH ran LoRA
supervised fine-tuning on LQH Cloud with:

| Setting | Value |
| --- | --- |
| Epochs | 3 |
| Optimizer steps | 2,250 |
| Micro-batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `5e-4` |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Maximum sequence length | 2,048 |
| Image tokens | 256 |
| Seed | 42 |

The cloud run completed in 79 minutes on one A100 40 GB GPU.

## Evaluation protocol

Checkpoint selection used a 400-image validation set. The final comparison
used a separate frozen test set of 400 images: 200 spiral and 200 elliptical.
The split was object-disjoint and exact-image-hash-disjoint from training. The
test set was excluded from training, manual review, and checkpoint selection.

Base and trained deployments received the same images, prompt, unconstrained
decoding configuration, and 32-token output limit. The primary metric parsed
the response deterministically and compared the exact label with the held-out
GZ2-derived target. A vision judge provided a secondary, advisory score.

## Results

| Frozen 400-image test | Base 450M | GZ2-trained 450M |
| --- | ---: | ---: |
| Exact label accuracy | 234/400 (58.5%) | 399/400 (99.75%) |
| Spiral recall | 34/200 (17.0%) | 199/200 (99.5%) |
| Elliptical recall | 200/200 (100%) | 200/200 (100%) |
| Strict output-format compliance | 400/400 | 400/400 |
| Advisory vision-judge mean | 7.79/10 | 9.98/10 |

The base model already returned the requested format, but it strongly favored
`elliptical` and missed most spiral examples. After post-training, that class
collapse was absent on the frozen test slice. The trained model made one error:
it predicted `elliptical` for GZ2 asset `125362`, whose held-out target was
`spiral`.

The observed gain followed several changes made together: a narrower task,
high-consensus human-vote labels, balanced classes, exact one-word targets, and
12,000 examples of LoRA SFT. This experiment did not run ablations, so it does
not establish how much improvement came from any one change.

## Scope of the result

The result demonstrates generalization within a high-consensus, in-distribution
Galaxy Zoo 2 test slice. It does not establish performance on ambiguous GZ2
objects, JWST or Rubin imagery, multiple-object detection, bounding-box
grounding, exact catalog identity, or physical galaxy typing. Those require
separate evaluation sets and, where needed, additional training.

The label-building implementation is in
[`pipelines/ingest/normalize_gz2.py`](../pipelines/ingest/normalize_gz2.py), the
balanced corpus selection is in
[`pipelines/ingest/prepare_gz2_balanced_12k.py`](../pipelines/ingest/prepare_gz2_balanced_12k.py),
and the evaluation rubric is in
[`evals/scorers/gz2_grounded_v1.md`](../evals/scorers/gz2_grounded_v1.md).
Raw datasets, cloud outputs, and operational run records remain outside version
control.

# Review synthetic image-to-morphology training targets

Score the assistant's morphology JSON against the attached CENTRAL galaxy image
and the schema supplied in the student's system instructions. Return
{"score": <integer 1-10>, "reasoning": "specific visible evidence and any violations"}.
This is an annotation-quality review, not an identity identification task.

First check exact JSON/schema and the following binding questionnaire rules.
Any malformed JSON, extra/missing key, invalid enum or branch violation MUST
score 1-2 regardless of whether the impossible field describes a real feature:

- roundness is applicable only when appearance=smooth.
- edge_on is applicable only when appearance=features_or_disk.
- bar, spiral and bulge_prominence are applicable only when
  appearance=features_or_disk AND edge_on=no.
- arms_winding and arms_number are applicable only when those parents apply
  AND spiral=yes. Every inapplicable field MUST be null.
- An applicable field may still be null because the image is ambiguous.
- In particular, a smooth galaxy MUST have null bulge_prominence, even with a
  bright core. An edge-on disk MUST have null bar, spiral, bulge_prominence and
  both arm fields. Do not invent additional branch dependencies.

Then assess supported non-null claims AND whether the response usefully
describes clearly visible features. Null means unresolved or inapplicable, not
absence. Do not penalize justified uncertainty in blurry or ambiguous images.
Do not award all-null a perfect score merely because it makes no false claim:
on an unmistakably thin edge-on disk, all-null misses the principal morphology
and should score at most 6. Inapplicable nulls are always correct.

Never guess a galaxy's catalog ID, distance, mass, age or physical type from
appearance. A bright core alone does not establish a dominant bulge. Smooth
elongation does not automatically establish a disk. Seeing a disk does not
automatically establish spiral arms. Judge only what THIS image supports.
In this Galaxy Zoo taxonomy, smooth refers to a smooth, unresolved light
distribution. A bright central concentration surrounded by a diffuse halo
can still be smooth. Neither a bright center nor a faint outer glow ALONE
establishes features_or_disk. That label needs resolved disk geometry or
structure such as a thin edge-on profile, bar, arm pattern, or discernible
structural features. Do not change smooth to features_or_disk solely because
you can see a center and surrounding light; do not confuse this visual label
with the object's unobserved physical galaxy type.

9-10: exact valid schema/branches, accurate main morphology and conservative
subfeatures, justified uncertainty. 7-8: useful/plausible morphology with minor
uncertainty. 4-6: substantial unsupported claim, meaningful visual conflict, or
excessive abstention despite clear main morphology. 1-3: severe hallucination
or wrong principal morphology; syntax/branch failures specifically 1-2.

Do not assume these synthetic targets are correct because they look polished.
Do not treat a prior label or judge score as ground truth. Explain the actual
image evidence. Deterministic code independently enforces schema and branches.

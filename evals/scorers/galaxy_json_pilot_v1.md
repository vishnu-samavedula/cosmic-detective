# Galaxy morphology JSON quality
Assess the assistant's JSON classification against the attached image and the system instructions. Return {"score": <1-10>, "reasoning": "brief specific evidence"}.

First parse the assistant text as JSON. A plain JSON object is correct; do NOT demand that it be wrapped inside a JSON string. Required top-level fields are schema_version (1) and features. features has appearance, roundness, edge_on, bar, spiral, bulge_prominence, arms_winding, arms_number. The allowed values and branch rules are in the system prompt.

Score 9-10: structurally and logically correct, non-null claims plausible in the image, appropriately cautious. Score 7-8: minor plausible disagreement or conservative missing information. Score 4-6: clear visible feature missed or questionable non-null claim. Score 1-3: malformed structure, branch contradiction, or clearly unsupported non-null details.

Null is a valid intentional abstention, including insufficient catalog-label agreement, not a refusal or schema error. All-null can be correct on ambiguous images. Do not penalize null on an inapplicable branch. Morphological descriptions such as smooth, spiral, and bar are NOT prohibited physical claims. One-armed galaxies are allowed; do not assert all spirals have two or more arms. Judge the image rather than treating a reference label as infallible. Never add requirements about prose, word count, catalog names, or unsupported certainty scores.

The labels were derived from human votes using conservative thresholds; slight discrepancies are not automatically critical errors. This judge is a supplemental visual audit; deterministic JSON/branch validation and catalog-vote comparisons are the primary measures.

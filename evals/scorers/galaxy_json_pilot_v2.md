# Audit catalog-derived morphology targets
You audit existing human-vote-derived JSON targets for a conservative classifier. Return {"score": <1-10>, "reasoning": "specific evidence"}.

The task is NOT to maximize the number of filled fields. Null means the human votes did not satisfy a support threshold OR the questionnaire branch is inapplicable. YOU DO NOT HAVE THE VOTE COUNTS. Therefore never deduct points for a null, including an entirely null answer. Do not infer that a null is wrong from image appearance. This conservative omission policy is intentional. Only assess NON-NULL claims for visual contradictions. Null does not assert that a feature is absent.

MANDATORY BRANCH RULES (these override general astronomical classification conventions):
- If appearance is smooth, roundness may be filled; edge_on, bar, spiral, bulge_prominence, arms_winding and arms_number MUST ALL BE NULL. A smooth galaxy's bright center MUST NOT be labeled dominant bulge in this schema.
- If appearance is features_or_disk, roundness MUST BE NULL, even if the image is round or elongated.
- If edge_on is yes OR null, bar, spiral, bulge_prominence, arms_winding and arms_number MUST BE NULL. An edge-on central bulge is deliberately not classified by the bulge_prominence question.
- Only when appearance=features_or_disk AND edge_on=no may bar, spiral and bulge_prominence be non-null.
- Only when spiral=yes may arms_winding and arms_number be non-null.
- Every feature may be null even when its branch is applicable.

Score 9-10: all non-null assertions plausible and rules satisfied; all-null is 10 because it asserts nothing unsupported. Score 7-8: a non-null assertion has minor visual uncertainty but is plausible. Score 4-6: a non-null assertion clearly conflicts with visible morphology. Score 1-3: invalid structure, branch violation, or major unsupported non-null claims. Do not require JSON to be a quoted string: an object is valid. One arm is an allowed category. Smooth/spiral/bar are morphology, NOT physical claims. Smooth refers to visual texture; elongated smooth does not prove elliptical or rule out an actual disk. Do not replace this scheme with your own.

Explain only actual non-null claims and rule violations. If no issue, say the non-null claims are plausible and nulls comply with the conservative policy. Deterministic code separately validates schema and branch logic.

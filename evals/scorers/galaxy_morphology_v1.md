# Scorer: Galaxy Morphology Vision QA (Cosmic Detective pilot v1)

You are judging a training sample for a vision-language model that answers
plain-English questions about a galaxy's visible morphology from a single
real galaxy image. The conversation shows the image as [image 1] followed by
the user's question, then the assistant's answer.

Judge the ASSISTANT'S ANSWER against the attached image and the question.

## Scoring scale (1-10)

- **9-10**: Answer is fully consistent with the image, directly addresses the
  question, reads as natural plain English for a curious non-expert, is 1-3
  sentences (~20-60 words), and calibrates uncertainty well — confident where
  the image clearly supports a claim, hedged ("appears", "may", "not clearly
  established") where the morphology is genuinely ambiguous.
- **7-8**: Good answer with minor issues — slightly long or short, slightly
  awkward wording, or a hedge that is a bit too strong/weak for what the image
  shows. Still grounded and useful.
- **4-6**: Mediocre — vague or generic ("this is a galaxy with stars"), misses
  the question's focus, noticeably over/under-confident relative to the image,
  or mildly jargon-y.
- **1-3**: Bad — one or more critical failures (below).

## Critical failures (score 1-3)

1. **Hallucinated morphology**: describes spiral arms, a bar, a dust lane, an
   edge-on disk, a ring, or any structure that is not plausibly present in
   [image 1]. Groundedness is THE failure mode for this task.
2. **Physical claims**: states or implies distance, size, mass, age,
   luminosity, redshift, exact catalog identity, or physical history — none of
   these are derivable from pixels. (Answering a distance/age question by
   saying it cannot be judged from the image, then describing what IS visible,
   is CORRECT, not a failure.)
3. **Label leakage**: mentions votes, percentages, statistics, Galaxy Zoo,
   catalogs, object IDs, surveys, or classification provenance.
4. **Not an answer**: refuses, evades, or answers something other than the
   question asked.

## Dimensions to weigh

- **Image consistency** (most important): every descriptive claim must be
  plausible in the attached image.
- **Question relevance**: the answer addresses what was asked.
- **Uncertainty calibration**: ambiguous-looking morphology gets qualitative
  hedges; clear morphology can be stated directly. Neither false confidence
  nor evasive hedging on an obvious galaxy.
- **Plain English**: accessible to a non-expert; no astronomical jargon
  (Hubble type codes, "debiased", "surface brightness", etc.).
- **Conciseness**: 1-3 sentences, roughly 20-60 words. Lists, headers, or
  markdown formatting are format violations (deduct ~2 points).

## Examples

**Good (9-10)** — question "What kind of galaxy is this?" on an elongated,
smooth, ambiguous galaxy:
"The galaxy appears elongated and mostly smooth. Spiral arms are not clearly
established in this image."
Grounded, hedged exactly right, plain, concise.

**Good (9)** — question "How far away is this galaxy?":
"I can't judge its distance from this image alone. What I can see is a
round, smooth-looking galaxy with a bright center and no clear spiral
structure."
Correctly declines the physical claim, then describes the visible.

**Mid (5)** — "This galaxy has beautiful spiral arms winding around a bright
core." on a smooth featureless blob: hallucinated arms → actually a critical
failure (2), not mid.

**Mid (5)** — "It is a galaxy." True but useless: vague, no observation,
no engagement with the question.

**Bad (2)** — "This is a barred spiral galaxy about 50,000 light-years across
in the SDSS catalog." Physical claims + label leakage.

Return a JSON object: {"score": <1-10>, "reasoning": "<2-3 sentences citing
what in the image supports or contradicts the answer>"}.

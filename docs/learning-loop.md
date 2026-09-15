# Learning-loop contract

Cosmic Detective currently visualizes a self-improving observatory, but it does
not submit training jobs. Feedback stays in browser local storage and the
progress bars are simulated. The repository skill at
[`skills/cosmic-learning-loop/SKILL.md`](../skills/cosmic-learning-loop/SKILL.md)
defines the intended handoff without implementing or executing it.

## Intended flow

```mermaid
flowchart TB
    Inbox[Reviewed feedback inbox]
    Export[Versioned feedback export<br/>image bytes or asset ref + SHA-256 + provenance]
    Morph[Pipeline A<br/>morphology label snapshot]
    Boxes[Pipeline B<br/>grounding box snapshot]
    MorphCheck[Label, conflict, dedup<br/>and eval-leakage checks]
    BoxCheck[Box, conflict, dedup<br/>and eval-leakage checks]
    Policy{Training policy satisfied?}
    Manual[Explicit snapshot approval]
    Auto[Configured automatic thresholds]
    Hygiene[LQH registration<br/>datamix + hygiene]
    Train[LQH start_training<br/>new candidate generation]
    Evaluate[Task-specific frozen evaluation]
    Promote{Promotion decision}
    Candidate[Candidate checkpoint]

    Inbox --> Export
    Export --> Morph --> MorphCheck --> Policy
    Export --> Boxes --> BoxCheck --> Policy
    Manual --> Policy
    Auto --> Policy
    Policy -->|yes| Hygiene --> Train --> Evaluate --> Promote
    Promote -->|accepted| Candidate

    classDef source fill:#10201b,stroke:#527d6b,color:#d8e7df
    classDef data fill:#101c22,stroke:#52717e,color:#dce8ec
    classDef gate fill:#241d13,stroke:#b08a50,color:#f0dfbd
    classDef cloud fill:#17172a,stroke:#706b9c,color:#e1def2
    class Inbox,Export source
    class Morph,Boxes,MorphCheck,BoxCheck data
    class Policy,Manual,Auto,Promote gate
    class Hygiene,Train,Evaluate,Candidate cloud
```

The two data paths are separate because they teach different outputs:

| Pipeline | Reviewed source | Training target | Candidate purpose |
| --- | --- | --- | --- |
| Morphology | Confirmed or corrected single-object classifications | Exact `spiral` or `elliptical` label | Improve central-galaxy morphology |
| Grounding | Confirmed or corrected multi-object maps | Reviewed normalized galaxy boxes | Improve multi-object localization |

Uncertain feedback remains available for human review but never becomes a
training target. The builders must transform accepted human annotations
deterministically; no VLM or text model fills missing labels or boxes.

## Missing implementation

Before this can run, the app needs an explicit feedback-export action or a
server-side ingest route. A training-grade export must add fields that the
current browser queue does not guarantee:

- schema version and stable feedback-record ID;
- original image bytes or durable immutable asset reference;
- explicit image SHA-256 and dimensions;
- task and source checkpoint version;
- original prediction and explicit reviewed target;
- reviewer or review-session provenance and timestamp.

Two deterministic builders must then validate and package the accepted records
as LQH VLM datasets. They should produce immutable manifests, rejection and
conflict reports, class or box distributions, and hashes of the exact rows sent
to training.

## Approval and automatic mode

Manual mode requires approval of a concrete dataset snapshot, not a changing
inbox. Automatic mode requires recorded thresholds for eligible example count,
class or scene coverage, conflict rate, validation success, evaluation
readiness, and budget. Reaching an inbox count by itself is not enough to start
training.

After either policy passes, LQH should register the dataset, build the datamix,
run contamination hygiene, and create a new candidate checkpoint from an
immutable parent. Training completion does not imply promotion. The candidate
must beat or match its parent on the frozen evaluation for the task it changes.

Morphology and grounding may eventually share one multi-task checkpoint. That
choice should wait until a single candidate can be evaluated against both the
existing GZ2 classification suite and a reviewed grounding suite, so gains in
one task cannot hide regressions in the other.

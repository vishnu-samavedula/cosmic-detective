# Repository layout

| Path | Purpose | Public repository |
| --- | --- | --- |
| `apps/web/client/` | Web interface and server-side inference proxy | Yes |
| `pipelines/ingest/` | Dataset validation and catalog builders | Yes |
| `pipelines/evaluate/` | Reproducible evaluation utilities | Yes |
| `data_gen/`, `prompts/`, `evals/` | Training-example and scoring definitions | Yes |
| `data/` | Raw and derived survey data | Placeholder documentation only |
| `datasets/`, `seed_data/` | Packaged model inputs | No |
| `runs/`, `training/` | Harness state, job records, and model outputs | No |
| `artifacts/`, `test/` | Generated reports, exports, and local test images | No |

The excluded directories may contain large files, local paths, remote job
identifiers, or third-party data whose source terms differ from the code license.

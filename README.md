# Cosmic Detective

Cosmic Detective is an experimental galaxy-morphology explorer. Choose a Galaxy
Zoo image or upload an observation, run it through either a base 450M vision
model or a morphology-tuned checkpoint, and compare the result. Successful
classifications can be saved as collectible field-journal cards with nearby
reference images.

The current classifier answers one deliberately narrow question: does the
central galaxy look **spiral** or **elliptical**? The answer describes visible
morphology. It does not identify a unique astronomical object, establish a
physical galaxy type, or replace scientific analysis.

## Features

- Explicit, user-triggered inference against base and tuned checkpoints
- Per-request latency, time-to-first-token, decode time, and token counts
- Galaxy Zoo 2 vote summaries for known reference objects
- Five morphology-filtered visual candidates for further inspection
- Browser-local field journal with JSON export
- Optional shuffle across a locally prepared 239,000-object catalog

Inference is connected through internal harness tooling. Public integration
guidance is coming soon; the server route currently expects an OpenAI-compatible
vision chat-completions endpoint.

## Models

Both sides of the comparison use
[LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), a
compact vision-language model with an LFM2.5-350M language backbone and an 86M
SigLIP2 NaFlex vision encoder.

| Mode | Model | Purpose |
| --- | --- | --- |
| Base 450M | Original LFM2.5-VL-450M checkpoint | Zero-shot morphology baseline |
| Trained 450M | LoRA adapter post-trained over the same checkpoint | Galaxy Zoo 2 spiral/elliptical classification |

The post-training corpus contains 12,000 balanced Galaxy Zoo 2 examples: 6,000
spiral and 6,000 elliptical. Each example pairs a galaxy image and classification
instruction with one expected lowercase label. The labels are grounded in the
top-level Galaxy Zoo 2 volunteer vote tree. The SFT recipe used three epochs, an
effective batch size of 16, a learning rate of `5e-4`, and LoRA rank 8.

On the same frozen, object-disjoint set of 400 images, the base model scored
7.79/10 and the post-trained model scored 9.98/10 using unconstrained decoding
and the same vision-judge protocol. These scores measure agreement with the
requested label and output format under this experiment. They are not calibrated
probabilities, scientific accuracy claims, or evidence that the model can
identify a unique catalog object.

The app connects to separately configured base and trained serving endpoints.
It does not ship or download either checkpoint. The model's own license and use
terms remain available on its model card.

## Architecture

```mermaid
flowchart LR
    A[Upload or choose a GZ2 image] --> B[Validate and resize in browser]
    B --> C[SHA-256 and 16x16 visual descriptor]
    B --> D[Server classification route]
    D --> E{Selected endpoint}
    E --> F[Base 450M]
    E --> G[GZ2-trained 450M]
    F --> H[spiral or elliptical]
    G --> H
    H --> I[Filter reference catalog by morphology]
    C --> J[Rank candidates by visual distance]
    I --> J
    J --> K[Five nearby GZ2 references]
    H --> L[Field-journal card]
    K --> L
```

The browser validates and downsizes the image before sending it through the
application server. The chosen model returns exactly `spiral` or `elliptical`.
That label filters a local Galaxy Zoo 2 reference catalog, and a small 16x16
grayscale descriptor ranks candidates within the matching morphology. An exact
SHA-256 match recognizes a reference image already present in the catalog.

The candidate ranker is a transparent image-distance heuristic, not a learned
embedding model. Its five results are visually similar references rather than
claims that the uploaded galaxy is the same astronomical object. Object IDs,
coordinates, vote summaries, and descriptions come from the reference catalog;
the vision model supplies the morphology label. Saved cards remain in browser
local storage.

## Run locally

Requirements: Node.js 22.13 or newer.

```sh
cd apps/web/client
npm install
cp .env.example .env.local
npm run dev
```

Set the four server-only variables in `.env.local`:

```dotenv
COSMIC_INFERENCE_BASE_URL=https://your-inference-host.example/v1
COSMIC_BASE_MODEL=your-base-model
COSMIC_TRAINED_MODEL=your-trained-model
COSMIC_INFERENCE_KEY=your-server-key
```

Never expose the inference key through a browser-prefixed environment variable.
Image uploads are resized in the browser and sent through the application server
to the configured inference endpoint. The app does not persist uploads;
collections stay in the browser's local storage.

## Data

Bulk survey data, generated catalogs, model weights, training runs, and local
experiment records are intentionally excluded from version control. After
obtaining the Galaxy Zoo 2 files, the scripts in `pipelines/ingest/` build the
local teaching catalog and full-archive browser index:

```sh
python3 pipelines/ingest/build_web_catalog.py
python3 pipelines/ingest/build_archive_pool.py
```

See [dataset notes](docs/datasets.md) for the expected local layout, source
attribution, and the distinction between source data and project code.

## Development

```sh
cd apps/web/client
npm run lint
npm run build
```

The web client lives in `apps/web/client/`. Data preparation and evaluation code
lives under `pipelines/`, `data_gen/`, `prompts/`, and `evals/`. Local datasets
and experiment outputs are covered by the root `.gitignore`.

## License

Project code is available under the [MIT License](LICENSE). Astronomy images,
catalogs, annotations, model checkpoints, and third-party assets retain their
respective source terms and are not relicensed by this repository.

## Image credit

The interface background is
[“NASA Reveals New Details About Dark Matter’s Influence on Universe”](https://www.flickr.com/photos/nasawebbtelescope/55118095230/)
from NASA’s James Webb Space Telescope photostream. Credit:
NASA/STScI/J. DePasquale/A. Pagan. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The app displays the
image with responsive cropping and a dark color overlay.

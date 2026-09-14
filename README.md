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

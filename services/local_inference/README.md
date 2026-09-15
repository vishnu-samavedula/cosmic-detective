# Local inference

This development server exposes the downloaded base 450M model and GZ2 LoRA
adapter through the OpenAI-compatible endpoint expected by the web app. It loads
one PEFT model and disables the adapter for base requests, avoiding two copies of
the base weights.

The local files are intentionally ignored by Git:

```text
models/LFM2.5-VL-450M-fc6221c/
models/gz2-450m-lora/
```

Create the isolated runtime once:

```sh
uv venv --python 3.11 .venv-local-inference
uv pip install --python .venv-local-inference/bin/python \
  'torch>=2.13,<2.15' 'torchvision>=0.28' \
  'transformers>=5.16,<5.17' 'peft>=0.20' \
  safetensors pillow fastapi 'uvicorn[standard]'
```

Start inference from the repository root:

```sh
.venv-local-inference/bin/uvicorn services.local_inference.server:app \
  --host 127.0.0.1 --port 8010
```

Then start a local-model copy of the web app:

```sh
cd apps/web/client
COSMIC_INFERENCE_BASE_URL=http://127.0.0.1:8010/v1 \
COSMIC_BASE_MODEL=local-base-450m \
COSMIC_TRAINED_MODEL=local-gz2-450m \
COSMIC_INFERENCE_KEY=local-development \
npm run dev -- --port 3004
```

Open `http://localhost:3004`. The first inference lazily loads the model and is
slower than subsequent requests. This server serializes generation requests and
is intended only for a single-user local demo.

"""Run an image-only smoke request; credentials never enter the saved report."""
import base64
import json
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
key_result = json.loads(Path('/tmp/cosmic-inference-key.json').read_text())
key = key_result['result']['secret']
assert isinstance(key, str) and key.startswith('lqh_inf_'), 'Missing inference key'
image = ROOT / 'data/raw/galaxy-zoo-2/kaggle-images/images_gz2/images/295305.jpg'
prompt = (ROOT / 'prompts/galaxy_morphology_v0.md').read_text()
question = 'Describe the central galaxy in this image. What structure can you see, and what is uncertain?'
payload = dict(model='cosmic-vision3b-v1', temperature=0, max_tokens=512, messages=[
    dict(role='system', content=prompt),
    dict(role='user', content=[dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(image.read_bytes()).decode())), dict(type='text', text=question)])])
started = time.monotonic()
with httpx.Client(timeout=300) as client:
    response = client.post('https://inference.lqh.ai/v1/chat/completions', headers={'Authorization': 'Bearer ' + key}, json=payload)
report = dict(image='295305.jpg', model=payload['model'], checkpoint='0d1f2eef-0ca3-4b5f-a062-de2fa3e20f0a', system_prompt=prompt, question=question, temperature=0, max_tokens=512, elapsed_seconds=round(time.monotonic()-started, 2), http_status=response.status_code)
try:
    report['response'] = response.json()
except ValueError:
    report['response'] = response.text[:2000]
out = ROOT / 'artifacts/evaluations/smoke-295305/checkpoint-api.json'
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

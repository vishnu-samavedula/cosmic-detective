const PROMPT =
  'Detect all instances of: galaxy. Inspect the entire frame before answering, including every quadrant and image edge. Favor recall: include faint, small, overlapping, partially cropped, and morphologically uncertain galaxy candidates. Use one tight box per distinct source and do not merge neighboring sources. Response must be only a JSON array in this exact shape: [{"label":"galaxy","bbox":[x1,y1,x2,y2]}]. Coordinates are normalized to [0,1].';

function inferenceConfig() {
  return {
    baseUrl:
      process.env.COSMIC_INFERENCE_BASE_URL ||
      process.env.LQH_INFERENCE_BASE_URL ||
      'https://inference.lqh.ai/v1',
    baseModel: process.env.COSMIC_BASE_MODEL || process.env.LQH_BASE_MODEL,
    key: process.env.COSMIC_INFERENCE_KEY || process.env.LQH_INFERENCE_KEY,
  };
}

async function readStream(response: Response, started: number) {
  if (!response.body)
    throw new Error('The inference service returned no data.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let content = '';
  let firstTokenAt: number | null = null;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (!data || data === '[DONE]') continue;
      const event = JSON.parse(data) as {
        choices?: Array<{ delta?: { content?: string } }>;
        error?: { message?: string };
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
        };
      };
      if (event.error) throw new Error('The inference stream failed.');
      const delta = event.choices?.[0]?.delta?.content || '';
      if (delta && firstTokenAt === null) firstTokenAt = Date.now();
      content += delta;
      if (event.usage) {
        inputTokens = event.usage.prompt_tokens ?? inputTokens;
        outputTokens = event.usage.completion_tokens ?? outputTokens;
      }
    }
    if (done) break;
  }
  const completedAt = Date.now();
  return {
    content,
    latencyMs: completedAt - started,
    ttftMs: firstTokenAt === null ? null : firstTokenAt - started,
    decodeMs: firstTokenAt === null ? null : completedAt - firstTokenAt,
    inputTokens,
    outputTokens,
  };
}

function parseDetections(raw: string) {
  const fenced = raw.match(/\[[\s\S]*\]/)?.[0];
  if (!fenced) throw new Error('The model returned no detection array.');
  const parsed = JSON.parse(fenced) as unknown;
  if (!Array.isArray(parsed))
    throw new Error('The model returned an invalid detection array.');

  return parsed
    .slice(0, 24)
    .map((item) => {
      if (!item || typeof item !== 'object') return null;
      const candidate = item as { label?: unknown; bbox?: unknown };
      if (
        !Array.isArray(candidate.bbox) ||
        candidate.bbox.length !== 4 ||
        !candidate.bbox.every(
          (value) => typeof value === 'number' && Number.isFinite(value),
        )
      )
        return null;
      const [rawX1, rawY1, rawX2, rawY2] = candidate.bbox as number[];
      const x1 = Math.max(0, Math.min(1, rawX1));
      const y1 = Math.max(0, Math.min(1, rawY1));
      const x2 = Math.max(0, Math.min(1, rawX2));
      const y2 = Math.max(0, Math.min(1, rawY2));
      if (x2 - x1 < 0.015 || y2 - y1 < 0.015) return null;
      return {
        label:
          typeof candidate.label === 'string'
            ? candidate.label.slice(0, 40)
            : 'galaxy',
        bbox: [x1, y1, x2, y2] as [number, number, number, number],
      };
    })
    .filter((item) => item !== null);
}

export async function POST(request: Request) {
  const config = inferenceConfig();
  const contentLength = Number(request.headers.get('content-length') || 0);
  if (contentLength > 8_100_000)
    return Response.json(
      { error: 'The scan request is too large.' },
      { status: 413 },
    );
  if (!config.key || !config.baseModel)
    return Response.json(
      { error: 'The base grounding endpoint is not configured.' },
      { status: 503 },
    );

  let body: { image?: unknown };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return Response.json(
      { error: 'The scan request is not valid JSON.' },
      { status: 400 },
    );
  }
  const image = typeof body.image === 'string' ? body.image : '';
  if (
    !/^data:image\/(jpeg|png|webp);base64,/.test(image) ||
    image.length > 8_000_000
  )
    return Response.json(
      { error: 'Send a JPEG, PNG, or WebP observation below 6 MB.' },
      { status: 400 },
    );

  const started = Date.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 150_000);
  try {
    const response = await fetch(
      `${config.baseUrl.replace(/\/$/, '')}/chat/completions`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${config.key}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: config.baseModel,
          temperature: 0,
          max_tokens: 512,
          stream: true,
          stream_options: { include_usage: true },
          messages: [
            {
              role: 'system',
              content:
                'You are an exhaustive visual grounding model. Search the full image and do not stop after finding the first valid source. Follow the requested JSON format exactly.',
            },
            {
              role: 'user',
              content: [
                { type: 'image_url', image_url: { url: image } },
                { type: 'text', text: PROMPT },
              ],
            },
          ],
        }),
        signal: controller.signal,
      },
    );
    if (!response.ok)
      throw new Error('The inference service could not complete detection.');
    const streamed = await readStream(response, started);
    const detections = parseDetections(streamed.content);
    return Response.json({
      detections,
      model: 'base',
      modelName: config.baseModel,
      latencyMs: streamed.latencyMs,
      ttftMs: streamed.ttftMs,
      decodeMs: streamed.decodeMs,
      inputTokens: streamed.inputTokens,
      outputTokens: streamed.outputTokens,
    });
  } catch (error) {
    console.error(
      'Detection request failed:',
      error instanceof Error ? error.name : 'Unknown error',
    );
    return Response.json(
      {
        error: controller.signal.aborted
          ? 'The grounding model took too long to wake. Try again.'
          : error instanceof SyntaxError
            ? 'The grounding model returned invalid JSON.'
            : error instanceof Error &&
                error.message.includes('detection array')
              ? error.message
              : 'The grounding model could not map this field.',
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}

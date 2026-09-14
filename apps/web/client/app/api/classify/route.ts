const PROMPT =
  'Classify the visible morphology of the central galaxy. Reply with exactly one lowercase label: spiral or elliptical.';

type ModelChoice = 'base' | 'trained';

function inferenceConfig() {
  return {
    baseUrl:
      process.env.COSMIC_INFERENCE_BASE_URL ||
      process.env.LQH_INFERENCE_BASE_URL ||
      'https://inference.lqh.ai/v1',
    baseModel: process.env.COSMIC_BASE_MODEL || process.env.LQH_BASE_MODEL,
    trainedModel:
      process.env.COSMIC_TRAINED_MODEL || process.env.LQH_TRAINED_MODEL,
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
  const ttftMs = firstTokenAt === null ? null : firstTokenAt - started;
  const decodeMs = firstTokenAt === null ? null : completedAt - firstTokenAt;
  return { content, ttftMs, decodeMs, inputTokens, outputTokens };
}

function configuration(choice: ModelChoice) {
  const config = inferenceConfig();
  if (choice === 'base') {
    return {
      model: config.baseModel,
      label: 'Base 450M',
    };
  }
  return {
    model: config.trainedModel,
    label: 'GZ2-trained 450M',
  };
}

export async function GET() {
  const config = inferenceConfig();
  return Response.json({
    base: Boolean(config.baseModel),
    trained: Boolean(config.trainedModel),
  });
}

export async function POST(request: Request) {
  const config = inferenceConfig();
  const contentLength = Number(request.headers.get('content-length') || 0);
  if (contentLength > 8_100_000) {
    return Response.json(
      { error: 'The scan request is too large.' },
      { status: 413 },
    );
  }

  if (!config.key) {
    return Response.json(
      { error: 'The observatory inference key is not configured.' },
      { status: 503 },
    );
  }

  let body: { image?: unknown; model?: unknown };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return Response.json(
      { error: 'The scan request is not valid JSON.' },
      { status: 400 },
    );
  }

  if (body.model !== 'base' && body.model !== 'trained') {
    return Response.json(
      { error: 'Choose a valid model before scanning.' },
      { status: 400 },
    );
  }
  const choice: ModelChoice = body.model;
  const image = typeof body.image === 'string' ? body.image : '';
  if (
    !/^data:image\/(jpeg|png|webp);base64,/.test(image) ||
    image.length > 8_000_000
  ) {
    return Response.json(
      { error: 'Send a JPEG, PNG, or WebP observation below 6 MB.' },
      { status: 400 },
    );
  }

  const selected = configuration(choice);
  if (!selected.model) {
    return Response.json(
      {
        error: `${selected.label} does not have a serving endpoint yet.`,
        code: 'model_unavailable',
      },
      { status: 503 },
    );
  }

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
          model: selected.model,
          temperature: 0,
          max_tokens: 8,
          stream: true,
          stream_options: { include_usage: true },
          messages: [
            {
              role: 'system',
              content:
                'You classify the visible morphology of the central galaxy. Follow the requested label format.',
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

    if (!response.ok) {
      console.error('Inference request failed with status', response.status);
      throw new Error('The inference service could not complete the scan.');
    }

    const streamed = await readStream(response, started);
    const raw = streamed.content.trim().toLowerCase();
    const label = raw.replace(/[.\s]+$/g, '');
    if (label !== 'spiral' && label !== 'elliptical') {
      return Response.json(
        {
          error: 'The model returned an invalid morphology label.',
          raw,
          code: 'invalid_output',
        },
        { status: 502 },
      );
    }

    const latencyMs = Date.now() - started;
    const tokensPerSecond =
      streamed.outputTokens !== null &&
      streamed.decodeMs !== null &&
      streamed.decodeMs > 0
        ? streamed.outputTokens / (streamed.decodeMs / 1000)
        : null;
    return Response.json({
      label,
      model: choice,
      modelName: selected.model,
      latencyMs,
      ttftMs: streamed.ttftMs,
      decodeMs: streamed.decodeMs,
      inputTokens: streamed.inputTokens,
      outputTokens: streamed.outputTokens,
      tokensPerSecond,
    });
  } catch (error) {
    console.error(
      'Inference request failed:',
      error instanceof Error ? error.name : 'Unknown error',
    );
    return Response.json(
      {
        error: controller.signal.aborted
          ? 'The model took too long to wake. Try the scan again.'
          : 'The inference service could not complete the scan.',
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}

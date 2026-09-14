'use client';

import Image from 'next/image';
import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Check,
  CheckCircle2,
  Circle,
  FlaskConical,
  Gauge,
  LoaderCircle,
  Play,
  ShieldCheck,
  Sparkles,
  Telescope,
  TriangleAlert,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

export type ModelChoice = 'base' | 'trained';

export type LabInference = {
  label: 'spiral' | 'elliptical';
  model: ModelChoice;
  modelName: string;
  latencyMs: number;
  ttftMs: number | null;
  decodeMs: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  tokensPerSecond: number | null;
};

export type LearningExample = {
  id: string;
  image: string;
  sourceName: string;
  task?: 'morphology' | 'grounding';
  prediction: string;
  humanLabel?: 'spiral' | 'elliptical';
  verdict:
    | 'confirmed'
    | 'corrected'
    | 'uncertain'
    | 'stress-failure'
    | 'grounding-confirmed'
    | 'grounding-corrected'
    | 'grounding-uncertain';
  transformation?: string;
  predictedBoxes?: Array<[number, number, number, number]>;
  correctedBoxes?: Array<[number, number, number, number]>;
  checkpoint: ModelChoice;
  createdAt: string;
};

type Observation = { image: string; name: string };
type Variant = {
  id: string;
  name: string;
  note: string;
};
type StressResult = Variant & {
  image: string;
  inference?: LabInference;
  error?: string;
};

const VARIANTS: Variant[] = [
  { id: 'original', name: 'Original', note: 'Reference signal' },
  { id: 'rotate', name: 'Rotated 90°', note: 'Orientation shift' },
  { id: 'dim', name: 'Dimmed', note: '45% brightness' },
  { id: 'noise', name: 'Sensor noise', note: 'Synthetic read noise' },
  { id: 'compress', name: 'Compressed', note: 'Low-bandwidth JPEG' },
  { id: 'crop', name: 'Center crop', note: '30% field removed' },
];

function duration(value: number | null | undefined) {
  if (typeof value !== 'number') return '—';
  return value < 1000
    ? `${Math.round(value)} ms`
    : `${(value / 1000).toFixed(1)} s`;
}

async function bitmapFor(image: string) {
  const response = await fetch(image);
  if (!response.ok) throw Error('The observation could not be opened.');
  return createImageBitmap(await response.blob());
}

async function transformObservation(image: string, variant: string) {
  const bitmap = await bitmapFor(image);
  const rotated = variant === 'rotate';
  const canvas = document.createElement('canvas');
  canvas.width = rotated ? bitmap.height : bitmap.width;
  canvas.height = rotated ? bitmap.width : bitmap.height;
  const context = canvas.getContext('2d');
  if (!context) {
    bitmap.close();
    throw Error('The image lab is unavailable in this browser.');
  }

  if (rotated) {
    context.translate(canvas.width, 0);
    context.rotate(Math.PI / 2);
    context.drawImage(bitmap, 0, 0);
  } else if (variant === 'crop') {
    const insetX = bitmap.width * 0.15;
    const insetY = bitmap.height * 0.15;
    context.drawImage(
      bitmap,
      insetX,
      insetY,
      bitmap.width * 0.7,
      bitmap.height * 0.7,
      0,
      0,
      canvas.width,
      canvas.height,
    );
  } else {
    if (variant === 'dim') context.filter = 'brightness(45%) contrast(110%)';
    context.drawImage(bitmap, 0, 0);
  }
  bitmap.close();

  if (variant === 'noise') {
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    for (let index = 0; index < pixels.data.length; index += 4) {
      const noise = (((index * 9301 + 49297) % 233280) / 233280 - 0.5) * 52;
      pixels.data[index] += noise;
      pixels.data[index + 1] += noise;
      pixels.data[index + 2] += noise;
    }
    context.putImageData(pixels, 0, 0);
  }

  return canvas.toDataURL('image/jpeg', variant === 'compress' ? 0.18 : 0.82);
}

export async function makeObservationPreview(image: string) {
  const bitmap = await bitmapFor(image);
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 260 / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext('2d')?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas.toDataURL('image/jpeg', 0.68);
}

async function infer(image: string, model: ModelChoice) {
  const response = await fetch('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image, model }),
  });
  const result = (await response.json()) as Partial<LabInference> & {
    error?: string;
  };
  if (!response.ok) throw Error(result.error || 'The model scan failed.');
  return result as LabInference;
}

export function StressLab({
  observation,
  model,
  available,
  onInvestigate,
  onQueue,
}: {
  observation: Observation | null;
  model: ModelChoice | null;
  available: boolean;
  onInvestigate: () => void;
  onQueue: (example: LearningExample) => void;
}) {
  const [results, setResults] = useState<StressResult[]>([]);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [message, setMessage] = useState('');
  const [selectedVariants, setSelectedVariants] = useState(
    VARIANTS.slice(1).map((variant) => variant.id),
  );
  const sequence = useMemo(
    () => [
      VARIANTS[0],
      ...VARIANTS.slice(1).filter((variant) =>
        selectedVariants.includes(variant.id),
      ),
    ],
    [selectedVariants],
  );

  const baseline = results.find(
    (result) => result.id === 'original',
  )?.inference;
  const completed = results.filter((result) => result.inference).length;
  const stable = baseline
    ? results.filter((result) => result.inference?.label === baseline.label)
        .length
    : 0;
  const stability =
    baseline && completed ? Math.round((stable / completed) * 100) : null;
  const labelFlips = baseline
    ? results.filter(
        (result) =>
          result.id !== 'original' &&
          result.inference &&
          result.inference.label !== baseline.label,
      )
    : [];

  async function run() {
    if (!observation || !model || !available) return;
    setRunning(true);
    setResults([]);
    setMessage('');
    for (let index = 0; index < sequence.length; index += 1) {
      const variant = sequence[index];
      setStep(index + 1);
      try {
        const image = await transformObservation(observation.image, variant.id);
        const inference = await infer(image, model);
        setResults((current) => [...current, { ...variant, image, inference }]);
      } catch (error) {
        setResults((current) => [
          ...current,
          {
            ...variant,
            image: observation.image,
            error: error instanceof Error ? error.message : 'The scan failed.',
          },
        ]);
      }
    }
    setRunning(false);
  }

  function updateSelection(next: string[]) {
    setSelectedVariants(next);
    setResults([]);
    setStep(0);
    setMessage(
      next.length
        ? 'Stress plan updated. Run the sequence when ready.'
        : 'Choose at least one stress category.',
    );
  }

  async function queueFailure(result: StressResult) {
    if (!result.inference || !baseline || !model || !observation) return;
    const preview = await makeObservationPreview(result.image);
    onQueue({
      id: `${Date.now()}-${result.id}`,
      image: preview,
      sourceName: observation.name,
      prediction: result.inference.label,
      humanLabel: baseline.label,
      verdict: 'stress-failure',
      transformation: result.name,
      checkpoint: model,
      createdAt: new Date().toISOString(),
    });
    setMessage(`${result.name} added to the learning queue.`);
  }

  async function queueAllLabelFlips() {
    for (const result of labelFlips) await queueFailure(result);
    if (labelFlips.length)
      setMessage(
        `${labelFlips.length} label flips added to the learning queue.`,
      );
  }

  return (
    <section className="lab-page">
      <div className="lab-hero">
        <div>
          <div className="eyebrow">ROBUSTNESS STATION / 02</div>
          <h1>
            Bend the signal.
            <br />
            <em>Test what holds.</em>
          </h1>
        </div>
        <p>
          Change the conditions around one observation and measure whether the
          model keeps the same morphology reading.
        </p>
      </div>

      {!observation || !model ? (
        <div className="lab-empty">
          <FlaskConical size={38} />
          <h2>Prepare an observation first.</h2>
          <p>
            Choose an image and a model in Investigate. Inference remains
            deliberate; entering the lab does not start a scan.
          </p>
          <Button onClick={onInvestigate}>
            Go to Investigate <ArrowRight size={16} />
          </Button>
        </div>
      ) : (
        <>
          <div className="stress-console">
            <div className="stress-source">
              <Image
                unoptimized
                width={424}
                height={424}
                src={observation.image}
                alt={observation.name}
              />
              <div>
                <span className="eyebrow">SOURCE OBSERVATION</span>
                <strong>{observation.name}</strong>
                <small>
                  {model === 'trained' ? 'GZ2-trained 450M' : 'Base 450M'}
                </small>
              </div>
            </div>
            <div className="stress-launch">
              <span className="eyebrow">CONTROLLED SEQUENCE</span>
              <h2>Six views. One morphology.</h2>
              <p>
                Orientation, light, sensor noise, compression, and framing are
                changed independently.
              </p>
              <Button
                className="primary-action"
                disabled={
                  running || !available || selectedVariants.length === 0
                }
                onClick={() => void run()}
              >
                {running ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Play size={16} />
                )}
                {running
                  ? `Scanning ${step} of ${sequence.length}`
                  : results.length
                    ? 'Run sequence again'
                    : 'Run stress sequence'}
              </Button>
              {!available && (
                <small>The selected endpoint is unavailable.</small>
              )}
            </div>
          </div>

          <fieldset className="stress-categories" disabled={running}>
            <div className="category-heading">
              <div>
                <legend>Choose stress categories</legend>
                <span>
                  The original reference always runs before selected tests.
                </span>
              </div>
              <div>
                <button
                  type="button"
                  onClick={() =>
                    updateSelection(
                      VARIANTS.slice(1).map((variant) => variant.id),
                    )
                  }
                >
                  Select all
                </button>
                <button type="button" onClick={() => updateSelection([])}>
                  Clear
                </button>
              </div>
            </div>
            <div className="category-grid">
              {VARIANTS.slice(1).map((variant) => (
                <label
                  aria-label={variant.name}
                  htmlFor={`stress-${variant.id}`}
                  key={variant.id}
                >
                  <input
                    id={`stress-${variant.id}`}
                    type="checkbox"
                    checked={selectedVariants.includes(variant.id)}
                    onChange={(event) =>
                      updateSelection(
                        event.target.checked
                          ? [...selectedVariants, variant.id]
                          : selectedVariants.filter((id) => id !== variant.id),
                      )
                    }
                  />
                  <span>
                    <strong>{variant.name}</strong>
                    <small>{variant.note}</small>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>

          {(running || results.length > 0) && (
            <div className="sequence-progress" aria-live="polite">
              <span
                style={{
                  width: `${running ? ((step - 1) / sequence.length) * 100 : 100}%`,
                }}
              />
            </div>
          )}

          {results.length > 0 && (
            <>
              <div className="stability-summary">
                <div>
                  <Gauge size={22} />
                  <span>STABILITY SCORE</span>
                  <strong>{stability === null ? '—' : `${stability}%`}</strong>
                </div>
                <p>
                  {baseline
                    ? `${stable} of ${completed} completed views retained the original classification. This measures consistency, not confidence.`
                    : `${completed} transformed views completed, but the original timed out. Run the sequence again to establish a stability reference.`}
                </p>
                {labelFlips.length ? (
                  <Button
                    variant="outline"
                    onClick={() => void queueAllLabelFlips()}
                  >
                    <Sparkles size={15} /> Add {labelFlips.length} label{' '}
                    {labelFlips.length === 1 ? 'flip' : 'flips'} to learning
                    queue
                  </Button>
                ) : (
                  <span className="no-label-flips">No label flips to add</span>
                )}
              </div>
              <div className="stress-grid">
                {results.map((result) => {
                  const changed = Boolean(
                    baseline &&
                    result.inference &&
                    result.inference.label !== baseline.label,
                  );
                  return (
                    <article className="stress-card" key={result.id}>
                      <div className="stress-card-image">
                        <Image
                          unoptimized
                          width={424}
                          height={424}
                          src={result.image}
                          alt={result.name}
                        />
                        <span data-changed={changed}>
                          {result.error
                            ? 'FAILED'
                            : !baseline
                              ? 'COMPLETED'
                              : changed
                                ? 'LABEL FLIP'
                                : 'STABLE'}
                        </span>
                      </div>
                      <div className="stress-card-copy">
                        <span className="eyebrow">{result.note}</span>
                        <h3>{result.name}</h3>
                        {result.inference ? (
                          <>
                            <strong>{result.inference.label}</strong>
                            <small>
                              {duration(result.inference.latencyMs)} total ·{' '}
                              {duration(result.inference.ttftMs)} TTFT
                            </small>
                          </>
                        ) : (
                          <p>{result.error}</p>
                        )}
                        {changed && (
                          <button onClick={() => void queueFailure(result)}>
                            Add label flip <ArrowRight size={13} />
                          </button>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            </>
          )}
          {message && (
            <output className="feedback">
              <Check size={16} /> {message}
            </output>
          )}
        </>
      )}
    </section>
  );
}

const PIPELINE_STAGES = [
  'Validate corrections',
  'Build dataset snapshot',
  'Submit cloud training',
  'Train LoRA adapter',
  'Run frozen evaluations',
  'Promote or reject',
];

export function LearningLoop({
  queue,
  onRemove,
  onInvestigate,
}: {
  queue: LearningExample[];
  onRemove: (id: string) => void;
  onInvestigate: () => void;
}) {
  const [policy, setPolicy] = useState<'manual' | 'automatic'>('manual');
  const [generationTwo, setGenerationTwo] = useState(68);
  const [generationThree, setGenerationThree] = useState(0);
  const [approved, setApproved] = useState(false);
  const reviewed = 74 + queue.length;
  const automaticallyApproved = policy === 'automatic' && reviewed >= 100;
  const trainingStarted = approved || automaticallyApproved;

  useEffect(() => {
    const timer = window.setInterval(
      () => setGenerationTwo((value) => (value >= 91 ? 68 : value + 1)),
      4200,
    );
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!trainingStarted) return;
    const timer = window.setInterval(
      () => setGenerationThree((value) => Math.min(36, value + 2)),
      1800,
    );
    return () => window.clearInterval(timer);
  }, [trainingStarted]);

  const counts = useMemo(
    () => ({
      corrected: queue.filter((item) => item.verdict === 'corrected').length,
      uncertain: queue.filter((item) => item.verdict === 'uncertain').length,
      hard: queue.filter((item) => item.verdict === 'stress-failure').length,
      confirmed: queue.filter(
        (item) =>
          item.verdict === 'confirmed' ||
          item.verdict === 'grounding-confirmed',
      ).length,
      grounding: queue.filter((item) => item.task === 'grounding').length,
    }),
    [queue],
  );

  return (
    <section className="lab-page learning-page">
      <div className="lab-hero">
        <div>
          <div className="eyebrow">OBSERVATORY SYSTEM / 03</div>
          <h1>
            Observe. Correct.
            <br />
            <em>Learn again.</em>
          </h1>
        </div>
        <p>
          Human review and robustness failures become a candidate dataset for
          the observatory&apos;s next checkpoint generation.
        </p>
      </div>

      <div className="loop-overview">
        <div className="queue-summary">
          <div className="eyebrow">LEARNING QUEUE / THIS BROWSER</div>
          <strong>{queue.length.toString().padStart(2, '0')}</strong>
          <span>new observations</span>
          <div className="queue-counts">
            <span>{counts.corrected} corrected</span>
            <span>{counts.hard} hard examples</span>
            <span>{counts.uncertain} uncertain</span>
            <span>{counts.confirmed} confirmed</span>
            <span>{counts.grounding} box annotations</span>
          </div>
        </div>
        <div className="policy-card">
          <div className="eyebrow">TRAINING POLICY</div>
          <h2>How should the next run begin?</h2>
          <fieldset className="policy-switch">
            <legend className="sr-only">Training policy</legend>
            <button
              type="button"
              data-active={policy === 'manual'}
              onClick={() => setPolicy('manual')}
            >
              Manual approval
            </button>
            <button
              type="button"
              data-active={policy === 'automatic'}
              onClick={() => setPolicy('automatic')}
            >
              Auto-start at 100
            </button>
          </fieldset>
          <div className="readiness-row">
            <span>{Math.min(reviewed, 100)} / 100 reviewed examples</span>
            <strong>{Math.min(reviewed, 100)}%</strong>
          </div>
          <div className="mini-progress">
            <span style={{ width: `${Math.min(reviewed, 100)}%` }} />
          </div>
          <Button
            className="primary-action"
            disabled={trainingStarted}
            onClick={() => {
              setApproved(true);
              setGenerationThree(6);
            }}
          >
            <ShieldCheck size={16} />
            {trainingStarted ? 'Snapshot approved' : 'Approve current snapshot'}
          </Button>
          <small>
            LQH <code>/train</code> is triggered according to the continual
            learning policy configured in the LQH console. This screen is a
            preview and does not submit a cloud job.
          </small>
        </div>
      </div>

      <section className="pipeline-panel">
        <div className="section-heading">
          <div>
            <div className="eyebrow">CHECKPOINT LINEAGE</div>
            <h2>Two generations in motion.</h2>
          </div>
          <span className="small">Base 450M → GZ2 Gen 01 → candidates</span>
        </div>
        <div className="lineage">
          <span>BASE 450M</span>
          <ArrowRight size={16} />
          <span>GZ2 GEN 01</span>
          <ArrowRight size={16} />
          <strong>GEN 02 · TRAINING</strong>
          <span className="lineage-branch">
            GEN 03 · {trainingStarted ? 'SUBMITTED' : 'STAGING'}
          </span>
        </div>

        <div className="run-grid">
          <TrainingRun
            generation="GENERATION 02"
            badge="SIMULATED TRAINING RUN"
            title="Hard-example adaptation"
            progress={generationTwo}
            stage={3}
            meta={['12,000 + hard cases', 'Epoch 2 of 3', 'LoRA rank 8']}
          />
          <TrainingRun
            generation="GENERATION 03"
            badge="LOCAL PIPELINE PREVIEW"
            title={
              trainingStarted
                ? 'Approved correction snapshot'
                : 'Correction snapshot'
            }
            progress={trainingStarted ? generationThree : reviewed}
            stage={trainingStarted ? 2 : 0}
            meta={[
              `${reviewed} reviewed`,
              `${queue.length} from this session`,
              policy === 'manual' ? 'Manual gate' : 'Automatic policy',
            ]}
            staging={!trainingStarted}
          />
        </div>
      </section>

      <section className="feedback-inbox">
        <div className="section-heading">
          <div>
            <div className="eyebrow">FEEDBACK INBOX</div>
            <h2>Evidence for the next generation.</h2>
          </div>
          {!queue.length && (
            <Button variant="outline" onClick={onInvestigate}>
              Review observations <ArrowRight size={15} />
            </Button>
          )}
        </div>
        {!queue.length ? (
          <div className="inbox-empty">
            <Telescope size={28} />
            <p>
              Confirm or correct a scan, or send a label flip from the Stress
              Lab. New evidence will appear here.
            </p>
          </div>
        ) : (
          <div className="inbox-grid">
            {queue.map((item) => (
              <article key={item.id}>
                <div className="inbox-preview">
                  <Image
                    unoptimized
                    width={260}
                    height={260}
                    src={item.image}
                    alt={item.sourceName}
                  />
                  {item.task === 'grounding' &&
                    (item.correctedBoxes || item.predictedBoxes || []).map(
                      ([x1, y1, x2, y2], index) => (
                        <span
                          key={`${item.id}-box-${index}`}
                          style={{
                            left: `${x1 * 100}%`,
                            top: `${y1 * 100}%`,
                            width: `${(x2 - x1) * 100}%`,
                            height: `${(y2 - y1) * 100}%`,
                          }}
                        />
                      ),
                    )}
                </div>
                <div>
                  <span className="eyebrow">
                    {item.verdict.replace('-', ' ')}
                  </span>
                  <h3>{item.sourceName}</h3>
                  <p>
                    {item.task === 'grounding'
                      ? `Model: ${item.predictedBoxes?.length || 0} boxes${item.correctedBoxes ? ` · target: ${item.correctedBoxes.length} boxes` : ''}`
                      : `Model: ${item.prediction}${item.humanLabel ? ` · target: ${item.humanLabel}` : ''}`}
                  </p>
                  {item.transformation && <small>{item.transformation}</small>}
                  <button onClick={() => onRemove(item.id)}>Remove</button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </section>
  );
}

function TrainingRun({
  generation,
  badge,
  title,
  progress,
  stage,
  meta,
  staging = false,
}: {
  generation: string;
  badge: string;
  title: string;
  progress: number;
  stage: number;
  meta: string[];
  staging?: boolean;
}) {
  return (
    <article className="training-run">
      <div className="run-top">
        <span className="eyebrow">{generation}</span>
        <span className="replay-badge">{badge}</span>
      </div>
      <h3>{title}</h3>
      <div className="run-progress-copy">
        <span>{staging ? 'Dataset readiness' : 'Pipeline progress'}</span>
        <strong>{Math.min(progress, 100)}%</strong>
      </div>
      <div className="run-progress">
        <span style={{ width: `${Math.min(progress, 100)}%` }} />
      </div>
      <div className="run-meta">
        {meta.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>
      <ol className="stage-list">
        {PIPELINE_STAGES.map((item, index) => {
          const complete = index < stage;
          const active = index === stage;
          return (
            <li data-active={active} data-complete={complete} key={item}>
              {complete ? (
                <CheckCircle2 size={15} />
              ) : active && !staging ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <Circle size={15} />
              )}
              <span>{item}</span>
              {active && <small>{staging ? 'waiting' : 'in progress'}</small>}
            </li>
          );
        })}
      </ol>
      <div className="run-signal">
        {staging ? <TriangleAlert size={15} /> : <Activity size={15} />}
        {staging
          ? 'Approval gate is holding this snapshot.'
          : 'Demo progress only; no cloud job is active.'}
      </div>
    </article>
  );
}

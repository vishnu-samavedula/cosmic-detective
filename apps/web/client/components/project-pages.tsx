import {
  ArrowDown,
  BookOpen,
  BoxSelect,
  CheckCircle2,
  Cloud,
  Database,
  Gauge,
  Image as ImageIcon,
  Laptop,
  ScanSearch,
  Sparkles,
} from 'lucide-react';

const TRAINING_FLOW = [
  {
    step: '01',
    title: 'Ground the labels',
    copy: 'Galaxy Zoo 2 volunteer votes select clear spiral and smooth-looking examples. Ambiguous records stay out.',
  },
  {
    step: '02',
    title: 'Balance the lesson',
    copy: '12,000 unique images: 6,000 spiral and 6,000 elliptical-looking. Every target is one exact word.',
  },
  {
    step: '03',
    title: 'Tune a small VLM',
    copy: 'LoRA adjusts a narrow set of weights in the 450M base model over three supervised epochs.',
  },
  {
    step: '04',
    title: 'Test unseen objects',
    copy: 'Base and tuned models receive the same prompt on a frozen, object-disjoint 400-image test set.',
  },
] as const;

export function ModelLabPage() {
  return (
    <section className="lab-page model-lab-page">
      <div className="lab-hero">
        <div>
          <div className="eyebrow">MODEL LAB / HOW IT LEARNS</div>
          <h1>
            Teach a small model
            <br />
            to see <em>one thing well.</em>
          </h1>
        </div>
        <p>
          One 450M vision-language model, one focused morphology task, and a
          direct comparison before and after post-training.
        </p>
      </div>

      <section className="model-scoreboard" aria-label="Held-out results">
        <div>
          <span>BASE 450M</span>
          <strong>58.5%</strong>
          <small>234 / 400 exact labels</small>
        </div>
        <ArrowDown aria-hidden="true" />
        <div className="trained-score">
          <span>GZ2-TUNED 450M</span>
          <strong>99.75%</strong>
          <small>399 / 400 exact labels</small>
        </div>
        <div className="score-detail">
          <span>SPIRAL RECALL</span>
          <strong>17% → 99.5%</strong>
          <small>Same frozen 200-image spiral slice</small>
        </div>
      </section>

      <section className="model-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">POST-TRAINING FLOW</div>
            <h2>Human votes become a focused visual lesson.</h2>
          </div>
          <span className="small">No synthetic labels in the successful run</span>
        </div>
        <div className="training-flow">
          {TRAINING_FLOW.map((item, index) => (
            <article key={item.step}>
              <span>{item.step}</span>
              <div>
                <h3>{item.title}</h3>
                <p>{item.copy}</p>
              </div>
              {index < TRAINING_FLOW.length - 1 && (
                <ArrowDown aria-hidden="true" />
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="model-split">
        <article className="data-shape-card">
          <div className="eyebrow">WHAT ONE TRAINING ROW LOOKS LIKE</div>
          <div className="message-row user-message">
            <ImageIcon size={18} />
            <div>
              <span>USER</span>
              <p>[galaxy image]</p>
              <p>Classify the central galaxy. Reply: spiral or elliptical.</p>
            </div>
          </div>
          <div className="message-row assistant-message">
            <Sparkles size={18} />
            <div>
              <span>EXPECTED ANSWER</span>
              <code>spiral</code>
            </div>
          </div>
          <small>
            The model learns a visual decision and an exact output contract.
            Catalog identity and descriptions stay outside the target.
          </small>
        </article>

        <article className="recipe-card">
          <div className="eyebrow">TUNING RECIPE</div>
          <dl>
            <div>
              <dt>Base</dt>
              <dd>LFM2.5-VL-450M</dd>
            </div>
            <div>
              <dt>Method</dt>
              <dd>LoRA SFT · rank 8</dd>
            </div>
            <div>
              <dt>Training</dt>
              <dd>3 epochs · batch 16</dd>
            </div>
            <div>
              <dt>Learning rate</dt>
              <dd>5 × 10⁻⁴</dd>
            </div>
            <div>
              <dt>Cloud run</dt>
              <dd>79 min · A100 40 GB</dd>
            </div>
            <div>
              <dt>Local form</dt>
              <dd>BF16 base + FP32 adapter</dd>
            </div>
          </dl>
        </article>
      </section>

      <section className="model-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">INFERENCE ARCHITECTURE</div>
            <h2>One interface, two ways to read the sky.</h2>
          </div>
        </div>
        <div className="runtime-flow">
          <div>
            <ImageIcon size={20} />
            <span>Observation</span>
          </div>
          <b>→</b>
          <div>
            <ScanSearch size={20} />
            <span>App API</span>
          </div>
          <b>→</b>
          <div className="runtime-choice">
            <span>
              <Laptop size={17} /> Local MPS
            </span>
            <span>
              <Cloud size={17} /> LQH Cloud
            </span>
          </div>
          <b>→</b>
          <div className="runtime-choice">
            <span>Base · zero-shot</span>
            <span>Base + LoRA</span>
          </div>
          <b>→</b>
          <div>
            <Gauge size={20} />
            <span>Label + timing</span>
          </div>
        </div>
        <p className="model-footnote">
          Multi-object mode first asks the base VLM for boxes, then sends each
          selected crop through the chosen morphology model. Similar catalog
          images are ranked afterward by the app.
        </p>
      </section>

      <section className="lab-boundaries">
        <article>
          <CheckCircle2 size={20} />
          <div>
            <h3>What the result shows</h3>
            <p>
              A compact VLM can learn this narrow, high-consensus GZ2 task and
              correct a strong zero-shot bias against spirals.
            </p>
          </div>
        </article>
        <article>
          <BoxSelect size={20} />
          <div>
            <h3>What remains open</h3>
            <p>
              Ambiguous galaxies, exact catalog identity, JWST transfer, and
              multi-object grounding each need their own evaluation.
            </p>
          </div>
        </article>
      </section>
    </section>
  );
}

export function AboutPage() {
  return (
    <section className="lab-page about-page">
      <div className="lab-hero">
        <div>
          <div className="eyebrow">ABOUT COSMIC DETECTIVE</div>
          <h1>
            A field journal for
            <br />
            <em>curious eyes.</em>
          </h1>
        </div>
        <p>
          A playful take on Galaxy Zoo: bring an image, compare what a small VLM
          sees, and leave behind better evidence for its next generation.
        </p>
      </div>

      <section className="about-missions">
        <article>
          <ScanSearch size={24} />
          <span>01 / INVESTIGATE</span>
          <h2>Read one galaxy.</h2>
          <p>
            Compare a zero-shot base with a GZ2-tuned checkpoint, inspect
            latency, and browse nearby morphology examples.
          </p>
        </article>
        <article>
          <BoxSelect size={24} />
          <span>02 / MAP</span>
          <h2>Explore a crowded field.</h2>
          <p>
            Ask the base VLM to propose galaxy regions, adjust its boxes, then
            classify each selected crop.
          </p>
        </article>
        <article>
          <Sparkles size={24} />
          <span>03 / TEACH</span>
          <h2>Turn correction into evidence.</h2>
          <p>
            Confirm labels, annotate boxes, stress the model, and preview a
            governed path toward another checkpoint.
          </p>
        </article>
      </section>

      <section className="about-principles">
        <div>
          <div className="eyebrow">THE IDEA</div>
          <h2>Small models can become useful specialists.</h2>
          <p>
            Cosmic Detective explores what happens when a compact visual model
            learns a clear scientific vocabulary and runs close to an
            observation—on a laptop today, and perhaps near a smart telescope
            tomorrow.
          </p>
        </div>
        <div className="principle-list">
          <span>
            <Database size={17} /> Human-grounded Galaxy Zoo 2 labels
          </span>
          <span>
            <Gauge size={17} /> Visible speed and throughput
          </span>
          <span>
            <BookOpen size={17} /> Results saved as a personal field journal
          </span>
        </div>
      </section>

      <section className="about-boundary">
        <div className="eyebrow">A CLEAR BOUNDARY</div>
        <p>
          The model describes visible spiral or smooth-looking morphology. It
          does not establish physical galaxy type or identify a unique object.
          Candidate images are visual references, not identity matches.
        </p>
      </section>

      <section className="about-sources">
        <div>
          <span>CLASSIFICATIONS</span>
          <a href="https://data.galaxyzoo.org/" target="_blank" rel="noreferrer">
            Galaxy Zoo 2 ↗
          </a>
        </div>
        <div>
          <span>BASE MODEL</span>
          <a
            href="https://huggingface.co/LiquidAI/LFM2.5-VL-450M"
            target="_blank"
            rel="noreferrer"
          >
            LFM2.5-VL-450M ↗
          </a>
        </div>
        <div>
          <span>BACKGROUND</span>
          <a
            href="https://www.flickr.com/photos/nasawebbtelescope/55118095230/"
            target="_blank"
            rel="noreferrer"
          >
            NASA Webb Telescope ↗
          </a>
        </div>
      </section>
    </section>
  );
}

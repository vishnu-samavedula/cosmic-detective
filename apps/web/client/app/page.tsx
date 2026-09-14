'use client';
import { useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import Link from 'next/link';
import Image from 'next/image';
import {
  ArrowUpRight,
  ArrowRight,
  Upload,
  Orbit,
  Scan,
  Plus,
  Check,
  Download,
  Telescope,
  X,
  Minus,
  BookOpen,
  RefreshCw,
} from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';

type Galaxy = {
  id: string;
  assetId: string;
  name: string;
  image: string;
  ra: number;
  dec: number;
  kind: string;
  description: string;
  hash: string;
  descriptor: number[];
  votes: {
    smooth: number;
    features: number;
    total: number;
    spiral: number;
    noSpiral: number;
  };
};
type Card = {
  id: string;
  title: string;
  image: string;
  galaxyId?: string;
  status: string;
  created: string;
  morphology?: 'spiral' | 'elliptical';
  model?: ModelChoice;
  candidateIds?: string[];
};
type Candidate = { galaxy: Galaxy; exact: boolean; distance: number };
type ModelChoice = 'base' | 'trained';
type Inference = {
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
const KEY = 'cosmic-detective-collection-v1';
const MORPHOLOGY_GUIDE = {
  spiral: {
    title: 'Spiral structure',
    summary:
      'Visible arms or curved structure around a central region place this observation on the featured or disk branch of the Galaxy Zoo decision tree.',
    lookFor:
      'Look for arm curvature, winding tightness, the number of arms, and whether a straight central bar crosses the core.',
  },
  elliptical: {
    title: 'Smooth appearance',
    summary:
      'A smooth light profile with no resolved arms is classified as elliptical-looking in this pilot. That describes the image appearance; it does not prove a physical galaxy type.',
    lookFor:
      'Compare roundness, elongation, central concentration, and whether faint disk structure could be hidden by image depth or resolution.',
  },
  unknown: {
    title: 'Outside the binary pilot',
    summary:
      'Artifacts, mergers, stars, weak signals, and ambiguous galaxies sit outside this checkpoint’s spiral-versus-elliptical contract.',
    lookFor:
      'Treat disagreement, invalid output, or poor image quality as a reason to inspect the source and seek broader labels rather than force an identity.',
  },
} as const;
function votePercent(value: number, total: number) {
  return total ? `${Math.round((value / total) * 100)}%` : '—';
}
function duration(value: number | null | undefined) {
  if (typeof value !== 'number') return '—';
  return value < 1000
    ? `${Math.round(value)} ms`
    : `${(value / 1000).toFixed(1)} s`;
}
export default function Home() {
  const [catalog, setCatalog] = useState<Galaxy[]>([]),
    [patches, setPatches] = useState<Galaxy[]>([]),
    [tab, setTab] = useState('investigate'),
    [active, setActive] = useState<Galaxy | null>(null),
    [detail, setDetail] = useState<Galaxy | null>(null),
    [journalDetail, setJournalDetail] = useState<Card | null>(null),
    [cards, setCards] = useState<Card[]>([]),
    [ready, setReady] = useState(false),
    [error, setError] = useState(''),
    [notice, setNotice] = useState(''),
    [upload, setUpload] = useState<{
      image: string;
      name: string;
      hash: string;
      descriptor: number[];
    } | null>(null),
    [candidates, setCandidates] = useState<Candidate[]>([]),
    [busy, setBusy] = useState(false),
    [scanning, setScanning] = useState(false),
    [shuffling, setShuffling] = useState(false),
    [modelChoice, setModelChoice] = useState<ModelChoice | null>(null),
    [modelStatus, setModelStatus] = useState({ base: false, trained: false }),
    [inference, setInference] = useState<Inference | null>(null),
    [limit, setLimit] = useState(5),
    [zoom, setZoom] = useState(1);
  const input = useRef<HTMLInputElement>(null);
  const library = useRef<HTMLDivElement>(null);
  const fullArchive = useRef<Galaxy[] | null>(null);
  useEffect(() => {
    fetch('/catalog.json')
      .then((r) => {
        if (!r.ok)
          throw Error(
            'The image library could not be loaded. Refresh to try again.',
          );
        return r.json();
      })
      .then((value) => {
        if (!Array.isArray(value) || !value.length)
          throw Error('The reference library is empty.');
        const c = value as Galaxy[];
        setCatalog(c);
        setPatches(c.slice(0, 12));
      })
      .catch((e) => setError(e.message));
    fetch('/api/classify')
      .then(
        async (r) => (await r.json()) as { base?: unknown; trained?: unknown },
      )
      .then((v) =>
        setModelStatus({ base: Boolean(v.base), trained: Boolean(v.trained) }),
      )
      .catch(() => {});
    queueMicrotask(() => {
      try {
        const v = JSON.parse(localStorage.getItem(KEY) || '[]');
        if (Array.isArray(v))
          setCards(
            v.filter(
              (x) =>
                x &&
                typeof x.id === 'string' &&
                typeof x.image === 'string' &&
                typeof x.title === 'string',
            ),
          );
      } catch {
        setError(
          'Your saved collection could not be read. New observations can still be collected.',
        );
      }
      setReady(true);
    });
  }, []);
  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(KEY, JSON.stringify(cards));
    } catch {
      queueMicrotask(() =>
        setError(
          'Browser storage is full or unavailable. Export your collection to keep it.',
        ),
      );
    }
  }, [cards, ready]);
  useEffect(() => {
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: unknown,
            options: { signal: AbortSignal },
          ) => void | Promise<void>;
        };
      }
    ).modelContext;
    if (!context?.registerTool || !catalog.length) return;
    const lifecycle = new AbortController();
    const register = (tool: unknown) => {
      try {
        void Promise.resolve(
          context.registerTool(tool, { signal: lifecycle.signal }),
        ).catch(() => {});
      } catch {
        /* Optional browser capability. */
      }
    };
    register({
      name: 'list_reference_galaxies',
      description:
        'Read the available reference galaxies; these are not inferred matches.',
      inputSchema: {
        type: 'object',
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      execute: () =>
        catalog.map((g) => ({ id: g.id, name: g.name, kind: g.kind })),
    });
    register({
      name: 'open_galaxy_field_notes',
      description:
        'Open the field notes for a known reference galaxy. Does not identify an upload or collect a card.',
      inputSchema: {
        type: 'object',
        properties: { objectId: { type: 'string' } },
        required: ['objectId'],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute: (input: unknown) => {
        if (
          !input ||
          typeof input !== 'object' ||
          !('objectId' in input) ||
          typeof input.objectId !== 'string'
        )
          throw Error('An objectId string is required.');
        const g = catalog.find((g) => g.id === input.objectId);
        if (!g) throw Error('Object is not in the reference library.');
        flushSync(() => setDetail(g));
        return { id: g.id, status: 'field_notes_open' };
      },
    });
    return () => lifecycle.abort();
  }, [catalog]);
  function collect(g: Galaxy | null) {
    const id = g?.id || upload?.hash;
    if (!id) return;
    if (cards.some((c) => c.id === id)) {
      if (!g && upload && inference) {
        const label =
          inference.label[0].toUpperCase() + inference.label.slice(1);
        setCards((items) =>
          items.map((card) =>
            card.id === id
              ? {
                  ...card,
                  title: `${label} morphology`,
                  status: `Morphology · ${inference.model === 'trained' ? 'trained' : 'base'} 450M`,
                  morphology: inference.label,
                  model: inference.model,
                  candidateIds: candidates.slice(0, 5).map((c) => c.galaxy.id),
                }
              : card,
          ),
        );
        setNotice('Saved card updated with the model classification.');
        return;
      }
      setNotice('Already in your collection.');
      return;
    }
    const status = g
      ? upload && !candidates.find((c) => c.galaxy.id === g.id)?.exact
        ? 'Possible match'
        : 'Archive-linked'
      : inference
        ? `Morphology · ${inference.model === 'trained' ? 'trained' : 'base'} 450M`
        : 'Unresolved';
    const inferredTitle = inference
      ? `${inference.label[0].toUpperCase() + inference.label.slice(1)} morphology`
      : 'Uncharted observation';
    setCards((c) => [
      {
        id,
        title: g?.kind || inferredTitle,
        image: g?.image || upload!.image,
        galaxyId: g?.id,
        status,
        created: new Date().toISOString(),
        morphology: g ? undefined : inference?.label,
        model: g ? undefined : inference?.model,
        candidateIds: g
          ? undefined
          : candidates.slice(0, 5).map((candidate) => candidate.galaxy.id),
      },
      ...c,
    ]);
    setNotice('Observation collected. Find it in your field journal.');
  }
  function choose(g: Galaxy) {
    setActive(g);
    setUpload(null);
    setModelChoice(null);
    setCandidates([]);
    setInference(null);
    setZoom(1);
    setNotice('');
    setTab('investigate');
  }
  async function shufflePatches() {
    setShuffling(true);
    setError('');
    try {
      if (!fullArchive.current) {
        const response = await fetch('/archive-pool.json.gz');
        if (!response.ok)
          throw Error('The full GZ2 archive index could not be loaded.');
        const rows = (await response.json()) as Array<
          [
            string,
            number,
            number,
            number,
            number,
            number,
            number,
            number,
            number,
            number,
          ]
        >;
        const kinds = [
          'Uncertain structure',
          'Smooth appearance',
          'Visible features',
          'Edge-on appearance',
          'Spiral structure',
        ];
        const descriptions = [
          'The classifications do not establish a single clear structure. A closer look or a sharper image could help.',
          'Most volunteers described a smooth-looking galaxy. Fine structure is not established by that classification.',
          'Most volunteers identified features or a disk rather than a simply smooth appearance.',
          'The votes favor a disk seen from the side. This orientation can make its overall structure harder to distinguish.',
          'The volunteers who reached the spiral question strongly favored visible spiral arms.',
        ];
        fullArchive.current = rows.map(
          ([
            id,
            assetId,
            ra,
            dec,
            kind,
            smooth,
            features,
            total,
            spiral,
            noSpiral,
          ]) => ({
            id,
            assetId: String(assetId),
            name: `SDSS ${id}`,
            image: `/gz2-all/${assetId}.jpg`,
            ra,
            dec,
            kind: kinds[kind],
            description: descriptions[kind],
            votes: { smooth, features, total, spiral, noSpiral },
            hash: `archive:${id}`,
            descriptor: [],
          }),
        );
      }
      const pool = fullArchive.current;
      const indexes = new Set<number>();
      while (indexes.size < 12)
        indexes.add(Math.floor(Math.random() * pool.length));
      setPatches(Array.from(indexes, (index) => pool[index]));
      setNotice(
        `12 new fields sampled from ${pool.length.toLocaleString()} grounded GZ2 galaxies.`,
      );
    } catch (e) {
      setError(
        e instanceof Error ? e.message : 'The archive could not be shuffled.',
      );
    } finally {
      setShuffling(false);
    }
  }
  function rankCandidates(
    descriptor: number[],
    hash: string,
    label?: 'spiral' | 'elliptical',
  ) {
    const matching = catalog.filter(
      (g) =>
        g.hash === hash ||
        !label ||
        (label === 'spiral'
          ? g.kind === 'Spiral structure'
          : g.kind === 'Smooth appearance'),
    );
    return matching
      .map((g) => ({
        galaxy: g,
        exact: g.hash === hash,
        distance: g.descriptor.reduce(
          (s, x, i) => s + (x - descriptor[i]) ** 2,
          0,
        ),
      }))
      .sort(
        (a, b) => Number(b.exact) - Number(a.exact) || a.distance - b.distance,
      )
      .slice(0, 10);
  }
  async function asInferenceImage(image: string) {
    if (image.startsWith('data:image/')) return image;
    const response = await fetch(image);
    if (!response.ok)
      throw Error('The selected archive image could not be loaded.');
    const blob = await response.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () =>
        typeof reader.result === 'string'
          ? resolve(reader.result)
          : reject(Error('The selected archive image could not be prepared.'));
      reader.onerror = () =>
        reject(Error('The selected archive image could not be prepared.'));
      reader.readAsDataURL(blob);
    });
  }
  async function imageDescriptor(image: string) {
    const blob = await (await fetch(image)).blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = 16;
    canvas.height = 16;
    const context = canvas.getContext('2d')!;
    context.drawImage(bitmap, 0, 0, 16, 16);
    bitmap.close();
    const pixels = context.getImageData(0, 0, 16, 16).data;
    const values = Array.from(
      { length: 256 },
      (_, index) =>
        (pixels[index * 4] * 0.299 +
          pixels[index * 4 + 1] * 0.587 +
          pixels[index * 4 + 2] * 0.114) /
        255,
    );
    const mean = values.reduce((sum, value) => sum + value, 0) / 256;
    const norm =
      Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0)) ||
      1;
    return values.map((value) => (value - mean) / norm);
  }
  async function classify(
    image = upload?.image || active?.image,
    hash = upload?.hash || active?.hash,
    descriptor = upload?.descriptor || active?.descriptor,
    choice = modelChoice,
  ) {
    if (!image || !hash || !descriptor) return;
    if (!choice) {
      setError('Choose the base or trained checkpoint before inference.');
      return;
    }
    setScanning(true);
    setError('');
    setNotice('');
    setInference(null);
    try {
      const inferenceImage = await asInferenceImage(image);
      const rankingDescriptor =
        descriptor.length === 256
          ? descriptor
          : await imageDescriptor(inferenceImage);
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: inferenceImage, model: choice }),
      });
      const result = (await response.json()) as Partial<Inference> & {
        error?: string;
      };
      if (!response.ok) throw Error(result.error || 'The model scan failed.');
      const found = result as Inference;
      setInference(found);
      setCandidates(rankCandidates(rankingDescriptor, hash, found.label));
      setNotice(
        `${found.label[0].toUpperCase() + found.label.slice(1)} morphology detected. Archive candidates updated.`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The model scan failed.');
      if (descriptor.length === 256)
        setCandidates(rankCandidates(descriptor, hash));
    } finally {
      setScanning(false);
    }
  }
  async function receive(file?: File) {
    if (!file) return;
    setError('');
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Choose a JPEG, PNG, or WebP image.');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError('Choose an image smaller than 15 MB.');
      return;
    }
    setBusy(true);
    setTab('investigate');
    setLimit(5);
    setZoom(1);
    try {
      const bytes = await file.arrayBuffer();
      const hash = Array.from(
        new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
      )
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
      const bitmap = await createImageBitmap(file);
      if (bitmap.width * bitmap.height > 40000000) {
        bitmap.close();
        throw Error('Choose an image below 40 megapixels.');
      }
      const canvas = document.createElement('canvas');
      canvas.width = 16;
      canvas.height = 16;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(bitmap, 0, 0, 16, 16);
      const pixels = ctx.getImageData(0, 0, 16, 16).data;
      const v = Array.from(
        { length: 256 },
        (_, i) =>
          (pixels[i * 4] * 0.299 +
            pixels[i * 4 + 1] * 0.587 +
            pixels[i * 4 + 2] * 0.114) /
          255,
      );
      const mean = v.reduce((a, b) => a + b, 0) / 256,
        norm = Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0)) || 1;
      const d = v.map((x) => (x - mean) / norm);
      const preview = document.createElement('canvas');
      const scale = Math.min(1, 1000 / Math.max(bitmap.width, bitmap.height));
      preview.width = Math.round(bitmap.width * scale);
      preview.height = Math.round(bitmap.height * scale);
      preview
        .getContext('2d')!
        .drawImage(bitmap, 0, 0, preview.width, preview.height);
      bitmap.close();
      const image = preview.toDataURL('image/jpeg', 0.85);
      setUpload({ image, name: file.name, hash, descriptor: d });
      setActive(null);
      setModelChoice(null);
      setInference(null);
      setCandidates(rankCandidates(d, hash));
      setNotice('Observation ready. Choose a model, then run inference.');
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'This image could not be decoded. Try another file.',
      );
    } finally {
      setBusy(false);
      if (input.current) input.current.value = '';
    }
  }
  function exportCollection() {
    const blob = new Blob(
      [
        JSON.stringify(
          { version: 1, source: 'Galaxy Zoo 2 / Hart 2016', cards },
          null,
          2,
        ),
      ],
      { type: 'application/json' },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cosmic-field-journal.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const shown = upload?.image || active?.image;
  return (
    <main className="observatory">
      <Tabs value={tab} onValueChange={(v) => setTab(String(v))}>
        <header className="masthead">
          <Link href="/" className="wordmark">
            <Orbit size={28} />
            <span>
              COSMIC
              <br />
              <b>DETECTIVE</b>
            </span>
          </Link>
          <TabsList className="main-tabs" variant="line">
            <TabsTrigger value="investigate">
              <Scan size={16} /> Investigate
            </TabsTrigger>
            <TabsTrigger value="collection">
              <BookOpen size={16} /> Collection{' '}
              <span className="count">
                {cards.length.toString().padStart(2, '0')}
              </span>
            </TabsTrigger>
          </TabsList>
          <span className="header-note">
            <i /> PERSONAL OBSERVATORY
          </span>
        </header>
        {error && (
          <div className="feedback error" role="alert">
            {error}
            <button aria-label="Dismiss error" onClick={() => setError('')}>
              <X size={16} />
            </button>
          </div>
        )}
        <TabsContent value="investigate">
          <section className="workspace">
            <div className="workspace-heading">
              <div>
                <div className="eyebrow">FIELD STATION / 01</div>
                <h1>
                  A closer look
                  <br />
                  at the <em>universe.</em>
                </h1>
              </div>
              <div className="intro">
                <p>
                  Bring a little piece of the sky.
                  <br />
                  See what you can discover.
                </p>
                <Button
                  className="upload-button"
                  onClick={() => input.current?.click()}
                  disabled={busy || !catalog.length}
                >
                  <Upload size={16} />
                  {busy ? 'Preparing image…' : 'Upload an observation'}
                </Button>
                <input
                  ref={input}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="sr-only"
                  aria-label="Upload a cosmic image"
                  onChange={(e) => receive(e.target.files?.[0])}
                />
                <span className="small">
                  JPEG, PNG, WebP · resized before LQH inference
                </span>
              </div>
            </div>
            <div className="investigation-grid">
              <div
                className="viewer"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  void receive(e.dataTransfer.files[0]);
                }}
              >
                <div className="viewer-top">
                  <span>
                    <i />
                    {upload ? 'YOUR OBSERVATION' : 'ARCHIVE TRANSMISSION'}
                  </span>
                  <span>{upload ? 'LQH VISION INPUT' : 'SDSS / OPTICAL'}</span>
                </div>
                <div className="view-image">
                  {shown ? (
                    <Image
                      unoptimized
                      width={424}
                      height={424}
                      src={shown}
                      alt={
                        upload
                          ? 'Your uploaded observation'
                          : `Galaxy ${active?.id}`
                      }
                      style={{ transform: `scale(${zoom})` }}
                    />
                  ) : (
                    <button
                      className="empty-viewer-action"
                      onClick={() =>
                        library.current?.scrollIntoView({ behavior: 'smooth' })
                      }
                    >
                      <Telescope size={28} />
                      <strong>No photo selected</strong>
                      <span>Upload one, or pick a patch of sky below.</span>
                    </button>
                  )}
                  <div className="reticle" aria-hidden="true" />
                </div>
                {shown && (
                  <div className="viewer-inference-action">
                    <Button
                      className="primary-action"
                      disabled={
                        scanning ||
                        busy ||
                        !modelChoice ||
                        !modelStatus[modelChoice]
                      }
                      onClick={() => void classify()}
                    >
                      <Scan size={16} />
                      {scanning
                        ? 'Running inference…'
                        : modelChoice
                          ? `Run inference · ${modelChoice === 'trained' ? 'trained' : 'base'} 450M`
                          : 'Choose a model to run inference'}
                    </Button>
                    {modelChoice && !modelStatus[modelChoice] && (
                      <span>Selected LQH endpoint is unavailable.</span>
                    )}
                  </div>
                )}
                <div className="viewer-bottom">
                  <span className="small">
                    {upload
                      ? upload.name
                      : `RA ${active?.ra.toFixed(4) || '—'}°  /  DEC ${active?.dec.toFixed(4) || '—'}°`}
                  </span>
                  <div className="zoom">
                    <button
                      aria-label="Zoom out"
                      disabled={zoom <= 1}
                      onClick={() => setZoom((z) => Math.max(1, z - 0.5))}
                    >
                      <Minus size={15} />
                    </button>
                    <span>{zoom.toFixed(1)}×</span>
                    <button
                      aria-label="Zoom in"
                      disabled={zoom >= 3}
                      onClick={() => setZoom((z) => Math.min(3, z + 0.5))}
                    >
                      <Plus size={15} />
                    </button>
                  </div>
                </div>
              </div>
              <aside className="readout">
                <div className="eyebrow">
                  {upload
                    ? scanning
                      ? 'MODEL SCAN IN PROGRESS'
                      : 'MODEL OBSERVATION'
                    : 'IN THE ARCHIVE'}
                </div>
                <h2>
                  {upload
                    ? scanning
                      ? 'Reading the signal…'
                      : inference
                        ? `${inference.label[0].toUpperCase() + inference.label.slice(1)} detected.`
                        : 'Ready to classify.'
                    : active?.kind || 'Looking for a signal'}
                </h2>
                <span className="status-badge">
                  {upload
                    ? inference
                      ? inference.model === 'trained'
                        ? 'GZ2 TRAINED · 450M'
                        : 'BASE · 450M'
                      : 'IDENTITY UNRESOLVED'
                    : 'KNOWN SOURCE IMAGE'}
                </span>
                <p>
                  {upload
                    ? inference
                      ? `The ${inference.model === 'trained' ? 'post-trained' : 'base'} vision model classified the visible central galaxy. This is a morphology reading, not an exact object identification.`
                      : 'Choose a model and scan the uploaded galaxy.'
                    : active?.description}
                </p>
                <div className="model-control">
                  <span>Vision model</span>
                  <RadioGroup
                    value={modelChoice || ''}
                    onValueChange={(value) => {
                      if (value) {
                        setModelChoice(value as ModelChoice);
                        setInference(null);
                        setNotice('Model selected. Run inference when ready.');
                      }
                    }}
                    className="model-slider"
                    aria-label="Vision model"
                  >
                    <label
                      htmlFor="model-base"
                      data-active={modelChoice === 'base'}
                      data-disabled={!modelStatus.base}
                    >
                      <RadioGroupItem
                        id="model-base"
                        value="base"
                        disabled={!modelStatus.base}
                      />
                      <span>
                        Base 450M
                        <small>
                          {modelStatus.base ? 'Zero-shot' : 'Endpoint needed'}
                        </small>
                      </span>
                    </label>
                    <label
                      htmlFor="model-trained"
                      data-active={modelChoice === 'trained'}
                      data-disabled={!modelStatus.trained}
                    >
                      <RadioGroupItem
                        id="model-trained"
                        value="trained"
                        disabled={!modelStatus.trained}
                      />
                      <span>
                        Trained 450M<small>GZ2 checkpoint</small>
                      </span>
                    </label>
                  </RadioGroup>
                  {shown && (
                    <Button
                      variant="outline"
                      className="scan-button"
                      disabled={
                        scanning ||
                        busy ||
                        !modelChoice ||
                        !modelStatus[modelChoice]
                      }
                      onClick={() => void classify()}
                    >
                      <Scan size={16} />
                      {scanning
                        ? 'Scanning…'
                        : modelChoice
                          ? `Scan with ${modelChoice === 'trained' ? 'trained' : 'base'} model`
                          : 'Choose a model first'}
                    </Button>
                  )}
                  {inference && (
                    <output className="inline-inference-result">
                      <span>MODEL RESULT</span>
                      <strong>
                        {inference.label[0].toUpperCase() +
                          inference.label.slice(1)}
                      </strong>
                      <small>
                        {inference.model === 'trained'
                          ? 'GZ2-trained 450M'
                          : 'Base 450M zero-shot'}{' '}
                        · {(inference.latencyMs / 1000).toFixed(1)}s
                      </small>
                      <div className="inference-metrics">
                        <span>
                          <b>{duration(inference.ttftMs)}</b>
                          Observed TTFT
                        </span>
                        <span>
                          <b>{duration(inference.decodeMs)}</b>
                          Decode span
                        </span>
                        <span>
                          <b>{duration(inference.latencyMs)}</b>
                          Total
                        </span>
                        <span>
                          <b>{inference.inputTokens ?? '—'}</b>
                          Input tokens
                        </span>
                        <span>
                          <b>{inference.outputTokens ?? '—'}</b>
                          Output tokens
                        </span>
                        <span>
                          <b>
                            {typeof inference.tokensPerSecond === 'number'
                              ? inference.tokensPerSecond.toFixed(1)
                              : '—'}
                          </b>
                          Decode tok/s
                        </span>
                      </div>
                      <em>
                        Timing observed by this app · token counts returned by
                        LQH Cloud
                      </em>
                    </output>
                  )}
                </div>
                <div className="readout-row">
                  <span>{upload ? 'Morphology' : 'Catalog identifier'}</span>
                  <strong>
                    {upload
                      ? inference
                        ? inference.label
                        : 'Awaiting inference'
                      : active?.name || '—'}
                  </strong>
                </div>
                <div className="readout-row">
                  <span>{upload ? 'Catalog identity' : 'Evidence'}</span>
                  <strong>
                    {upload
                      ? 'Not established'
                      : 'Galaxy Zoo volunteer classifications'}
                  </strong>
                </div>
                <div className="readout-actions">
                  <Button
                    className="primary-action"
                    disabled={!shown || busy || scanning}
                    onClick={() => collect(upload ? null : active)}
                  >
                    <Plus size={16} />
                    Collect observation
                  </Button>
                  {!upload && active && (
                    <Button variant="ghost" onClick={() => setDetail(active)}>
                      Read the field notes <ArrowUpRight size={16} />
                    </Button>
                  )}
                </div>
                <div className="model-note">
                  <span className="signal-dot" />{' '}
                  {upload ? 'Inference via LQH Cloud' : 'Archive record'}
                  <br />
                  <span>
                    {upload
                      ? 'The resized observation is sent to the selected endpoint.'
                      : 'Descriptions come from Galaxy Zoo classifications.'}
                  </span>
                </div>
              </aside>
            </div>
            {notice && (
              <output className="feedback" aria-live="polite">
                <Check size={16} />
                {notice}
              </output>
            )}
            {upload && (
              <section className="candidate-section">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">
                      {inference ? `${inference.label.toUpperCase()} / ` : ''}
                      {catalog.length} REFERENCE IMAGES
                    </div>
                    <h2>
                      {inference
                        ? 'Morphology-matched candidates'
                        : 'Closest visual candidates'}
                    </h2>
                  </div>
                  <span className="small">
                    {inference
                      ? 'Model category, then visual ranking'
                      : 'Visual ranking while inference is pending'}
                  </span>
                </div>
                <div className="candidate-grid">
                  {candidates.slice(0, limit).map((c, i) => (
                    <button
                      className="candidate"
                      key={c.galaxy.id}
                      onClick={() => setDetail(c.galaxy)}
                    >
                      <div>
                        <Image
                          unoptimized
                          width={424}
                          height={424}
                          src={c.galaxy.image}
                          alt={c.galaxy.kind}
                        />
                        <span>{String(i + 1).padStart(2, '0')}</span>
                      </div>
                      <strong>{c.galaxy.kind}</strong>
                      <span>
                        {c.exact
                          ? 'Exact archive file'
                          : 'Possible visual relative'}
                      </span>
                      <small>{c.galaxy.name}</small>
                    </button>
                  ))}
                </div>
                {limit === 5 && (
                  <Button variant="outline" onClick={() => setLimit(10)}>
                    Show five more <ArrowRight size={16} />
                  </Button>
                )}
                <p className="small">
                  These are category-matched examples, not claims of catalog
                  identity.
                </p>
              </section>
            )}
            {upload && inference?.model === 'trained' && (
              <section className="morphology-brief">
                <div>
                  <div className="eyebrow">
                    TRAINED CHECKPOINT / INTERPRETATION
                  </div>
                  <h2>{MORPHOLOGY_GUIDE[inference.label].title}</h2>
                  <p>{MORPHOLOGY_GUIDE[inference.label].summary}</p>
                  <strong>What to inspect next</strong>
                  <p>{MORPHOLOGY_GUIDE[inference.label].lookFor}</p>
                </div>
                <aside>
                  <span>How the five candidates were found</span>
                  <ol>
                    <li>
                      The trained 450M checkpoint assigns the upload to spiral
                      or elliptical-looking morphology.
                    </li>
                    <li>The archive is filtered to that Galaxy Zoo branch.</li>
                    <li>
                      A compact grayscale image descriptor ranks visual
                      similarity.
                    </li>
                  </ol>
                  <p>
                    Select a candidate to open its detailed field record: SDSS
                    identifier, sky coordinates, volunteer vote counts, and
                    classification notes.
                  </p>
                  <div className="unknown-note">
                    <b>{MORPHOLOGY_GUIDE.unknown.title}</b>
                    <span>{MORPHOLOGY_GUIDE.unknown.summary}</span>
                  </div>
                </aside>
              </section>
            )}
            <section className="library" ref={library}>
              <div className="section-heading">
                <div>
                  <div className="eyebrow">NO PHOTO? START HERE</div>
                  <h2>Pick a patch of sky.</h2>
                </div>
                <div className="archive-heading-actions">
                  <span className="small">
                    Real galaxies. Your next observation.
                  </span>
                  <Button
                    variant="outline"
                    onClick={() => void shufflePatches()}
                    disabled={shuffling}
                  >
                    <RefreshCw size={15} />
                    {shuffling ? 'Opening archive…' : 'Shuffle 12'}
                  </Button>
                </div>
              </div>
              <div className="archive-strip">
                {patches.map((g, i) => (
                  <button
                    key={g.id}
                    className={`archive-item ${active?.id === g.id && !upload ? 'selected' : ''}`}
                    onClick={() => choose(g)}
                  >
                    <div>
                      <Image
                        unoptimized
                        width={424}
                        height={424}
                        src={g.image}
                        alt={g.kind}
                        loading="lazy"
                      />
                      <span>{String(i + 1).padStart(2, '0')}</span>
                    </div>
                    <strong>{g.kind}</strong>
                    <span>SDSS · {g.assetId}</span>
                  </button>
                ))}
              </div>
            </section>
          </section>
        </TabsContent>
        <TabsContent value="collection">
          <section className="collection">
            <div className="section-heading">
              <div>
                <div className="eyebrow">YOUR PERSONAL FIELD JOURNAL</div>
                <h1>
                  A universe
                  <br />
                  <em>of your own.</em>
                </h1>
              </div>
              <Button
                variant="outline"
                disabled={!cards.length}
                onClick={exportCollection}
              >
                <Download size={16} /> Export journal
              </Button>
            </div>
            <p className="collection-note">
              {cards.length} observations · saved on this browser
            </p>
            {!cards.length ? (
              <div className="empty-collection">
                <Telescope size={40} />
                <h2>Your first discovery is waiting.</h2>
                <p>
                  Collect an archive object or keep an unanswered observation.
                </p>
                <Button onClick={() => setTab('investigate')}>
                  Back to the sky <ArrowRight size={16} />
                </Button>
              </div>
            ) : (
              <div className="card-grid">
                {cards.map((c, i) => (
                  <article className="specimen" key={c.id}>
                    <div className="card-top">
                      <span>
                        CD / {String(cards.length - i).padStart(3, '0')}
                      </span>
                      <span>{c.status}</span>
                    </div>
                    <button
                      className="specimen-image"
                      onClick={() => {
                        const g = catalog.find((g) => g.id === c.galaxyId);
                        if (g) setDetail(g);
                        else if (c.candidateIds?.length) setJournalDetail(c);
                        else
                          setNotice(
                            'This observation has no established catalog identity yet.',
                          );
                      }}
                      aria-label={`Read ${c.title}`}
                    >
                      <Image
                        unoptimized
                        width={424}
                        height={424}
                        src={c.image}
                        alt={c.title}
                      />
                      <ArrowUpRight size={20} />
                    </button>
                    <div className="specimen-info">
                      <div className="eyebrow">
                        {c.galaxyId
                          ? 'GALAXY OBSERVATION'
                          : c.status.startsWith('Morphology')
                            ? 'MODEL-CLASSIFIED OBSERVATION'
                            : 'UNRESOLVED OBSERVATION'}
                      </div>
                      <h2>{c.title}</h2>
                      <span className="small">
                        {c.galaxyId
                          ? 'SDSS ' + c.galaxyId
                          : c.status.startsWith('Morphology')
                            ? 'Morphology classified · catalog identity remains open'
                            : 'Identity remains open'}
                      </span>
                      <div className="card-bottom">
                        <time>{new Date(c.created).toLocaleDateString()}</time>
                        <button
                          onClick={() => {
                            setCards((cs) => cs.filter((x) => x.id !== c.id));
                            setNotice('Observation removed from collection.');
                          }}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
            {notice && (
              <output aria-live="polite" className="feedback">
                {notice}
              </output>
            )}
          </section>
        </TabsContent>
        <footer>
          <span>
            <Orbit size={14} /> COSMIC DETECTIVE
          </span>
          <span>Images: SDSS / Galaxy Zoo 2 · Labels: Hart et al. 2016</span>
          <a
            href="https://data.galaxyzoo.org/"
            target="_blank"
            rel="noreferrer"
          >
            Data & attribution <ArrowUpRight size={14} />
          </a>
          <a
            href="https://www.flickr.com/photos/nasawebbtelescope/55118095230/"
            target="_blank"
            rel="noreferrer"
          >
            Background: NASA/STScI/J. DePasquale/A. Pagan · CC BY 4.0{' '}
            <ArrowUpRight size={14} />
          </a>
        </footer>
      </Tabs>
      <Sheet
        open={!!detail}
        onOpenChange={(open) => {
          if (!open) setDetail(null);
        }}
      >
        <SheetContent className="field-sheet w-full sm:max-w-xl overflow-y-auto">
          <SheetHeader>
            <div className="eyebrow">FIELD NOTES / GALAXY ZOO 2</div>
            <SheetTitle className="text-2xl mt-3">{detail?.kind}</SheetTitle>
            <SheetDescription>{detail?.name}</SheetDescription>
          </SheetHeader>
          {detail && (
            <div className="detail-body">
              <Image
                unoptimized
                width={424}
                height={424}
                src={detail.image}
                alt={detail.kind}
              />
              <h3>What the classifications tell us</h3>
              <p>{detail.description}</p>
              <div className="vote-grid">
                <div>
                  <strong>
                    {votePercent(detail.votes.smooth, detail.votes.total)}
                  </strong>
                  <span>Smooth share · {detail.votes.smooth} votes</span>
                </div>
                <div>
                  <strong>
                    {votePercent(detail.votes.features, detail.votes.total)}
                  </strong>
                  <span>Features / disk · {detail.votes.features} votes</span>
                </div>
                <div>
                  <strong>{detail.votes.total}</strong>
                  <span>Top-level volunteer votes</span>
                </div>
                <div>
                  <strong>{detail.votes.spiral}</strong>
                  <span>Spiral-arm votes on eligible branches</span>
                </div>
              </div>
              <p className="small">
                Raw volunteer votes, not model confidence. Follow-up questions
                were answered only on eligible branches.
              </p>
              <h3>What to look for</h3>
              <p>
                {detail.kind === 'Spiral structure'
                  ? MORPHOLOGY_GUIDE.spiral.lookFor
                  : MORPHOLOGY_GUIDE.elliptical.lookFor}
              </p>
              <p className="small">
                This record describes morphology in the SDSS image. Similar
                appearance alone does not establish that an uploaded image is
                this same catalog object.
              </p>
              <h3>Where it is</h3>
              <p>
                RA {detail.ra.toFixed(5)}°<br />
                Dec {detail.dec.toFixed(5)}°
              </p>
              <p className="small">
                Identifier and coordinates come from the joined archive. Common
                names, distances, and additional catalog facts have not been
                retrieved.
              </p>
              <Button
                className="primary-action"
                onClick={() => collect(detail)}
              >
                {cards.some((c) => c.id === detail.id) ? (
                  <Check size={16} />
                ) : (
                  <Plus size={16} />
                )}{' '}
                {cards.some((c) => c.id === detail.id)
                  ? 'In your collection'
                  : 'Collect observation'}
              </Button>
              <p className="small">{notice}</p>
              <a
                href="https://data.galaxyzoo.org/"
                target="_blank"
                rel="noreferrer"
              >
                Classification source <ArrowUpRight size={14} />
              </a>
            </div>
          )}
        </SheetContent>
      </Sheet>
      <Sheet
        open={!!journalDetail}
        onOpenChange={(open) => {
          if (!open) setJournalDetail(null);
        }}
      >
        <SheetContent className="field-sheet w-full sm:max-w-2xl overflow-y-auto">
          <SheetHeader>
            <div className="eyebrow">SAVED OBSERVATION / MODEL RESULT</div>
            <SheetTitle className="text-2xl mt-3">
              {journalDetail?.title}
            </SheetTitle>
            <SheetDescription>{journalDetail?.status}</SheetDescription>
          </SheetHeader>
          {journalDetail && (
            <div className="detail-body journal-observation">
              <Image
                unoptimized
                width={424}
                height={424}
                src={journalDetail.image}
                alt={journalDetail.title}
              />
              <h3>What the checkpoint recognized</h3>
              <p>
                {journalDetail.morphology
                  ? MORPHOLOGY_GUIDE[journalDetail.morphology].summary
                  : 'This observation was saved before a morphology result was available.'}
              </p>
              <p className="small">
                The morphology result is saved. The catalog identity remains
                open because visual similarity does not establish that two
                images show the same astronomical object.
              </p>
              <h3>Five visual relatives</h3>
              <div className="journal-candidates">
                {(journalDetail.candidateIds || []).map((id, index) => {
                  const galaxy = catalog.find((item) => item.id === id);
                  if (!galaxy) return null;
                  return (
                    <button
                      key={id}
                      onClick={() => {
                        setJournalDetail(null);
                        setDetail(galaxy);
                      }}
                    >
                      <Image
                        unoptimized
                        width={424}
                        height={424}
                        src={galaxy.image}
                        alt={galaxy.kind}
                      />
                      <span>{String(index + 1).padStart(2, '0')}</span>
                      <strong>{galaxy.name}</strong>
                      <small>{galaxy.kind}</small>
                    </button>
                  );
                })}
              </div>
              <p className="small">
                Select a relative for its Galaxy Zoo votes, SDSS identifier,
                coordinates, and field notes.
              </p>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </main>
  );
}

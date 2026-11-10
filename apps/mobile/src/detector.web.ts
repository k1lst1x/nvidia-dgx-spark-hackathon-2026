import type { DetectionResult, LocalDetection } from './detector.types';

type OrtTensor = { data: Float32Array };
type OrtSession = { run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>> };
type OrtRuntime = {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  InferenceSession: { create: (url: string, options: object) => Promise<OrtSession> };
  Tensor: new (type: string, data: Float32Array, dimensions: number[]) => unknown;
};

const INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.4;
const IOU_THRESHOLD = 0.45;
const MAX_RESULTS = 12;
const MAX_FRAME_COVERAGE = 0.96;
const PREDICTIONS = 8400;
const LABEL_CONFIDENCE_FLOORS: Record<string, number> = {
  // These rounded COCO classes are frequent false positives for glasses, cans,
  // lids, and other objects held close to a face.
  frisbee: 0.75,
  'sports ball': 0.70,
};
const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];

// Real-world physical vertical height priors (in meters) for optical distance estimation
const OBJECT_HEIGHT_PRIORS: Record<string, number> = {
  person: 1.70,
  bicycle: 1.00,
  car: 1.50,
  motorcycle: 1.10,
  bus: 3.10,
  train: 3.80,
  truck: 2.80,
  traffic_light: 1.20,
  fire_hydrant: 0.70,
  stop_sign: 0.80,
  bench: 0.85,
  cat: 0.30,
  dog: 0.55,
  horse: 1.60,
  backpack: 0.45,
  umbrella: 0.85,
  handbag: 0.35,
  suitcase: 0.65,
  bottle: 0.24,
  wine_glass: 0.20,
  cup: 0.12,
  fork: 0.18,
  knife: 0.20,
  spoon: 0.16,
  bowl: 0.12,
  banana: 0.18,
  apple: 0.08,
  sandwich: 0.10,
  chair: 0.85,
  couch: 0.85,
  potted_plant: 0.45,
  bed: 0.75,
  dining_table: 0.75,
  tv: 0.55,
  laptop: 0.30,
  mouse: 0.06,
  remote: 0.18,
  keyboard: 0.15,
  cell_phone: 0.15,
  microwave: 0.35,
  oven: 0.75,
  sink: 0.60,
  refrigerator: 1.70,
  book: 0.24,
  clock: 0.30,
};

function estimateSpatialInfo(item: LocalDetection): { distance_m: number; direction: 'left' | 'ahead' | 'right' } {
  const normPrior = OBJECT_HEIGHT_PRIORS[item.label.replace(/\s+/g, '_')] ?? 0.50;
  // Normalized focal length constant for typical webcam vertical FOV (~58 deg)
  const fNorm = 1.15;
  const rawDist = (normPrior * fNorm) / Math.max(0.04, item.height);
  const distance_m = Math.round(Math.max(0.3, Math.min(8.0, rawDist)) * 10) / 10;

  const centerX = item.x + item.width / 2;
  const direction: 'left' | 'ahead' | 'right' = centerX < 0.35 ? 'left' : centerX > 0.65 ? 'right' : 'ahead';

  return { distance_m, direction };
}

type TrackState = LocalDetection & {
  missed: number;
  id: number;
  prevDistance?: number;
  lastSeenTime: number;
};

/**
 * Production-grade Temporal Object Tracker with EMA coordinate smoothing,
 * approach velocity estimation, and hysteresis confidence persistence.
 */
class TemporalObjectTracker {
  private tracks: TrackState[] = [];
  private nextId = 1;
  private readonly emaAlpha = 0.68; // Smooth, low-latency filter
  private readonly matchIouThreshold = 0.28;

  update(rawDetections: LocalDetection[]): LocalDetection[] {
    const matchedTrackIndices = new Set<number>();
    const matchedDetectionIndices = new Set<number>();
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();

    // 1. Greedy IoU matching with label constraint
    for (let d = 0; d < rawDetections.length; d++) {
      const candidate = rawDetections[d];
      let bestIou = this.matchIouThreshold;
      let bestTrackIndex = -1;

      for (let t = 0; t < this.tracks.length; t++) {
        if (matchedTrackIndices.has(t)) continue;
        const track = this.tracks[t];
        if (track.label === candidate.label) {
          const overlap = iou(track, candidate);
          if (overlap > bestIou) {
            bestIou = overlap;
            bestTrackIndex = t;
          }
        }
      }

      if (bestTrackIndex !== -1) {
        matchedTrackIndices.add(bestTrackIndex);
        matchedDetectionIndices.add(d);
        const track = this.tracks[bestTrackIndex];

        // Smooth bounding box coordinates with EMA filter
        track.x = this.emaAlpha * candidate.x + (1 - this.emaAlpha) * track.x;
        track.y = this.emaAlpha * candidate.y + (1 - this.emaAlpha) * track.y;
        track.width = this.emaAlpha * candidate.width + (1 - this.emaAlpha) * track.width;
        track.height = this.emaAlpha * candidate.height + (1 - this.emaAlpha) * track.height;
        track.confidence = this.emaAlpha * candidate.confidence + (1 - this.emaAlpha) * track.confidence;

        const spatial = estimateSpatialInfo(track);
        const dt = Math.max(0.05, (now - track.lastSeenTime) / 1000);
        if (track.prevDistance != null && spatial.distance_m != null) {
          const dDist = track.prevDistance - spatial.distance_m;
          const speed = dDist / dt; // positive means object is getting closer
          track.velocity_mps = Math.round(speed * 10) / 10;
          track.is_approaching = speed > 0.35 && spatial.distance_m <= 2.5 && spatial.direction === 'ahead';
        } else {
          track.is_approaching = false;
          track.velocity_mps = 0;
        }

        track.prevDistance = spatial.distance_m;
        track.lastSeenTime = now;
        track.distance_m = spatial.distance_m;
        track.direction = spatial.direction;
        track.missed = 0;
      }
    }

    // 2. Add new tracks for unmatched detections
    for (let d = 0; d < rawDetections.length; d++) {
      if (!matchedDetectionIndices.has(d)) {
        const item = rawDetections[d];
        const spatial = estimateSpatialInfo(item);
        this.tracks.push({
          ...item,
          distance_m: spatial.distance_m,
          direction: spatial.direction,
          id: this.nextId++,
          missed: 0,
          prevDistance: spatial.distance_m,
          lastSeenTime: now,
          is_approaching: false,
          velocity_mps: 0,
        });
      }
    }

    // 3. Age unmatched existing tracks (Hysteresis persistence to prevent flickering)
    for (let t = 0; t < this.tracks.length; t++) {
      if (!matchedTrackIndices.has(t)) {
        this.tracks[t].missed += 1;
        this.tracks[t].confidence *= 0.88;
      }
    }

    // Keep active tracks (survive up to 2 missed frames if previously confident)
    this.tracks = this.tracks.filter((t) => t.missed <= 2 && t.confidence >= 0.22);

    // 4. Priority Sort: Approaching hazards first, then person, then closest obstacles (<1.5m), then highest confidence
    const sorted = [...this.tracks].sort((a, b) => {
      if (a.is_approaching && !b.is_approaching) return -1;
      if (b.is_approaching && !a.is_approaching) return 1;
      if (a.label === 'person' && b.label !== 'person') return -1;
      if (b.label === 'person' && a.label !== 'person') return 1;
      const distA = a.distance_m ?? 5;
      const distB = b.distance_m ?? 5;
      if (Math.abs(distA - distB) > 0.8) return distA - distB;
      return b.confidence - a.confidence;
    });

    return sorted.map(({ missed: _m, prevDistance: _p, lastSeenTime: _l, ...item }) => item);
  }

  clear() {
    this.tracks = [];
  }
}

const tracker = new TemporalObjectTracker();

function assetUrl(path: string) {
  const base = document.querySelector('script[src*="/_expo/"]')?.getAttribute('src')?.split('/_expo/')[0] || '';
  return `${base}/${path}`.replace(/([^:]\/)\/+/, '$1');
}

let runtimePromise: Promise<OrtRuntime> | null = null;
let sessionPromise: Promise<OrtSession> | null = null;

function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = new Promise((resolve, reject) => {
      const existing = (globalThis as typeof globalThis & { ort?: OrtRuntime }).ort;
      if (existing) return resolve(existing);
      const script = document.createElement('script');
      script.src = assetUrl('ort/ort.wasm.min.js');
      script.onload = () => {
        const loaded = (globalThis as typeof globalThis & { ort?: OrtRuntime }).ort;
        if (loaded) resolve(loaded);
        else reject(new Error('ONNX Runtime did not initialise'));
      };
      script.onerror = () => reject(new Error('ONNX Runtime could not be loaded'));
      document.head.appendChild(script);
    });
  }
  return runtimePromise;
}

function getSession() {
  if (!sessionPromise) {
    sessionPromise = getRuntime().then((ort) => {
      ort.env.wasm.wasmPaths = assetUrl('ort/');
      ort.env.wasm.numThreads = globalThis.crossOriginIsolated
        ? Math.min(4, navigator.hardwareConcurrency || 1)
        : 1;
      // Ultralytics YOLO11n, COCO, 640 input, output [1, 84, 8400]. AGPL-3.0.
      return ort.InferenceSession.create(assetUrl('models/yolo11n.onnx'), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    });
  }
  return sessionPromise;
}

function loadImage(uri: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Camera frame could not be decoded'));
    image.src = uri;
  });
}

function iou(a: LocalDetection, b: LocalDetection) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  return intersection / (a.width * a.height + b.width * b.height - intersection);
}

function suppressOverlaps(items: LocalDetection[]) {
  const kept: LocalDetection[] = [];
  for (const candidate of [...items].sort((a, b) => b.confidence - a.confidence)) {
    const overlaps = kept.some((existing) => {
      const overlap = iou(existing, candidate);
      if (existing.label === candidate.label && overlap >= IOU_THRESHOLD) return true;
      if (overlap >= 0.78) return true;
      return false;
    });
    if (!overlaps) {
      kept.push(candidate);
      if (kept.length === MAX_RESULTS) break;
    }
  }
  return kept;
}

function focusOnObjects(items: LocalDetection[]) {
  const reliable = items.filter((item) => item.confidence >= (LABEL_CONFIDENCE_FLOORS[item.label] ?? SCORE_THRESHOLD));
  return suppressOverlaps(reliable);
}

function mapBoxToSource(
  centerX: number,
  centerY: number,
  boxWidth: number,
  boxHeight: number,
  scale: number,
  padX: number,
  padY: number,
  sourceWidth: number,
  sourceHeight: number,
): Omit<LocalDetection, 'label' | 'confidence'> | null {
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const left = (centerX - boxWidth / 2 - padX) / renderedWidth;
  const top = (centerY - boxHeight / 2 - padY) / renderedHeight;
  const right = (centerX + boxWidth / 2 - padX) / renderedWidth;
  const bottom = (centerY + boxHeight / 2 - padY) / renderedHeight;
  const x = Math.max(0, left);
  const y = Math.max(0, top);
  const width = Math.min(1, right) - x;
  const height = Math.min(1, bottom) - y;
  if (width < 0.02 || height < 0.02 || width * height > MAX_FRAME_COVERAGE) return null;
  return { x, y, width, height };
}

export async function detectObjects(uri: string): Promise<DetectionResult> {
  try {
    const [ort, session, image] = await Promise.all([getRuntime(), getSession(), loadImage(uri)]);
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Image preprocessing is unavailable');
    // Preserve the camera's aspect ratio. Stretching a portrait/landscape frame
    // into a square makes YOLOX predict broad, inaccurate boxes.
    const scale = Math.min(INPUT_SIZE / image.naturalWidth, INPUT_SIZE / image.naturalHeight);
    const renderedWidth = image.naturalWidth * scale;
    const renderedHeight = image.naturalHeight * scale;
    const padX = (INPUT_SIZE - renderedWidth) / 2;
    const padY = (INPUT_SIZE - renderedHeight) / 2;
    context.fillStyle = 'rgb(114, 114, 114)';
    context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    context.drawImage(image, padX, padY, renderedWidth, renderedHeight);
    const rgba = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const plane = INPUT_SIZE * INPUT_SIZE;
    const input = new Float32Array(plane * 3);
    for (let pixel = 0; pixel < plane; pixel += 1) {
      const offset = pixel * 4;
      input[pixel] = rgba[offset] / 255;
      input[plane + pixel] = rgba[offset + 1] / 255;
      input[plane * 2 + pixel] = rgba[offset + 2] / 255;
    }
    const output = await session.run({ images: new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]) });
    const values = output.output0.data as Float32Array;
    const detections: LocalDetection[] = [];
    for (let index = 0; index < PREDICTIONS; index += 1) {
      let classIndex = 0;
      let confidence = 0;
      for (let label = 0; label < COCO.length; label += 1) {
        const score = values[(4 + label) * PREDICTIONS + index];
        if (score > confidence) {
          confidence = score;
          classIndex = label;
        }
      }
      if (confidence < SCORE_THRESHOLD) continue;
      const centerX = values[index];
      const centerY = values[PREDICTIONS + index];
      const width = values[PREDICTIONS * 2 + index];
      const height = values[PREDICTIONS * 3 + index];
      const box = mapBoxToSource(
        centerX, centerY, width, height, scale, padX, padY, image.naturalWidth, image.naturalHeight,
      );
      if (!box) continue;
      detections.push({
        label: COCO[classIndex], confidence, ...box,
      });
    }
    const filtered = focusOnObjects(detections);
    const tracked = tracker.update(filtered);
    return { available: true, detections: tracked };
  } catch (error) {
    console.warn('Local YOLO11 inference unavailable', error);
    return { available: false, detections: [] };
  }
}

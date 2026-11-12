import { Asset } from 'expo-asset';
import { decode } from 'jpeg-js';
import { analyzeBlockedView } from './blockedView';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import type { DetectionResult, LocalDetection } from './detector.types';

const INPUT_SIZE = 416;
const SCORE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45;
// Keep enough independent objects for a busy room. The downstream UI and
// speech summary still prioritize the nearest/highest-confidence results.
const MAX_RESULTS = 20;
const MAX_FRAME_COVERAGE = 0.82;
const LABEL_CONFIDENCE_FLOORS: Record<string, number> = {
  frisbee: 0.72,
  'sports ball': 0.66,
};
const MODEL = require('../assets/models/yolox_nano.onnx');
const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];

const GRID: Array<{ x: number; y: number; stride: number }> = [];
for (const stride of [8, 16, 32]) {
  const size = INPUT_SIZE / stride;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) GRID.push({ x, y, stride });
  }
}

let sessionPromise: Promise<InferenceSession> | null = null;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const asset = Asset.fromModule(MODEL);
      await asset.downloadAsync();
      if (!asset.localUri) throw new Error('YOLOX model asset is unavailable');
      return InferenceSession.create(asset.localUri, {
        executionProviders: ['xnnpack', 'cpu'],
        logSeverityLevel: 3,
      });
    })();
  }
  return sessionPromise;
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
  const sorted = [...items].sort((a, b) => b.confidence - a.confidence);
  const kept: LocalDetection[] = [];
  for (const candidate of sorted) {
    if (kept.every((existing) => existing.label !== candidate.label || iou(existing, candidate) < IOU_THRESHOLD)) {
      kept.push(candidate);
      if (kept.length === MAX_RESULTS) break;
    }
  }
  return kept;
}

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
  const fNorm = 1.15;
  const rawDist = (normPrior * fNorm) / Math.max(0.04, item.height);
  const distance_m = Math.round(Math.max(0.3, Math.min(8.0, rawDist)) * 10) / 10;
  const centerX = item.x + item.width / 2;
  const direction: 'left' | 'ahead' | 'right' =
    centerX < 0.35 ? 'left' : centerX > 0.65 ? 'right' : 'ahead';
  return { distance_m, direction };
}

type TrackState = LocalDetection & {
  missed: number;
  id: number;
  prevDistance?: number;
  lastSeenTime: number;
};

class TemporalObjectTracker {
  private tracks: TrackState[] = [];
  private nextId = 1;
  private readonly emaAlpha = 0.68;
  private readonly matchIouThreshold = 0.28;

  update(rawDetections: LocalDetection[]): LocalDetection[] {
    const matchedTrackIndices = new Set<number>();
    const matchedDetectionIndices = new Set<number>();
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();

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

        track.x = this.emaAlpha * candidate.x + (1 - this.emaAlpha) * track.x;
        track.y = this.emaAlpha * candidate.y + (1 - this.emaAlpha) * track.y;
        track.width = this.emaAlpha * candidate.width + (1 - this.emaAlpha) * track.width;
        track.height = this.emaAlpha * candidate.height + (1 - this.emaAlpha) * track.height;
        track.confidence = this.emaAlpha * candidate.confidence + (1 - this.emaAlpha) * track.confidence;

        const spatial = estimateSpatialInfo(track);
        const dt = Math.max(0.05, (now - track.lastSeenTime) / 1000);
        if (track.prevDistance != null && spatial.distance_m != null) {
          const dDist = track.prevDistance - spatial.distance_m;
          const speed = dDist / dt;
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

    for (let t = 0; t < this.tracks.length; t++) {
      if (!matchedTrackIndices.has(t)) {
        this.tracks[t].missed += 1;
        this.tracks[t].confidence *= 0.88;
      }
    }

    this.tracks = this.tracks.filter((t) => t.missed <= 2 && t.confidence >= 0.22);

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

function isLikelySelfPerson(item: LocalDetection) {
  return item.label === 'person' && item.width >= 0.7 && item.x <= 0.16 && item.y <= 0.28;
}

function focusOnObjects(items: LocalDetection[]) {
  const reliable = items.filter((item) => item.confidence >= (LABEL_CONFIDENCE_FLOORS[item.label] ?? SCORE_THRESHOLD));
  const nonSelf = reliable.filter((item) => !isLikelySelfPerson(item));
  const nonPeople = nonSelf.filter((item) => item.label !== 'person');
  return suppressOverlaps(nonPeople.length ? nonPeople : nonSelf);
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
  if (width < 0.025 || height < 0.025 || width * height > MAX_FRAME_COVERAGE) return null;
  return { x, y, width, height };
}

function letterbox(image: ReturnType<typeof decode>) {
  const scale = Math.min(INPUT_SIZE / image.width, INPUT_SIZE / image.height);
  const renderedWidth = image.width * scale;
  const renderedHeight = image.height * scale;
  const padX = (INPUT_SIZE - renderedWidth) / 2;
  const padY = (INPUT_SIZE - renderedHeight) / 2;
  const plane = INPUT_SIZE * INPUT_SIZE;
  const input = new Float32Array(plane * 3);
  for (let y = 0; y < INPUT_SIZE; y += 1) {
    for (let x = 0; x < INPUT_SIZE; x += 1) {
      const pixel = y * INPUT_SIZE + x;
      const sourceX = Math.floor((x - padX) / scale);
      const sourceY = Math.floor((y - padY) / scale);
      if (sourceX < 0 || sourceY < 0 || sourceX >= image.width || sourceY >= image.height) {
        input[pixel] = 114;
        input[plane + pixel] = 114;
        input[plane * 2 + pixel] = 114;
        continue;
      }
      const rgba = (sourceY * image.width + sourceX) * 4;
      input[pixel] = image.data[rgba];
      input[plane + pixel] = image.data[rgba + 1];
      input[plane * 2 + pixel] = image.data[rgba + 2];
    }
  }
  return { input, scale, padX, padY };
}

export async function detectObjects(uri: string): Promise<DetectionResult> {
  try {
    const [session, response] = await Promise.all([getSession(), fetch(uri)]);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const image = decode(bytes, { useTArray: true, formatAsRGBA: true });
    const { input, scale, padX, padY } = letterbox(image);
    const output = await session.run({ images: new Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]) });
    const values = output.output.data as Float32Array;
    const detections: LocalDetection[] = [];
    for (let row = 0; row < 3549; row += 1) {
      const offset = row * 85;
      const objectness = values[offset + 4];
      if (objectness < SCORE_THRESHOLD) continue;
      let classIndex = 0;
      let classScore = 0;
      for (let index = 0; index < 80; index += 1) {
        if (values[offset + 5 + index] > classScore) {
          classScore = values[offset + 5 + index];
          classIndex = index;
        }
      }
      const confidence = objectness * classScore;
      if (confidence < SCORE_THRESHOLD) continue;
      const grid = GRID[row];
      const centerX = (values[offset] + grid.x) * grid.stride;
      const centerY = (values[offset + 1] + grid.y) * grid.stride;
      const width = Math.exp(values[offset + 2]) * grid.stride;
      const height = Math.exp(values[offset + 3]) * grid.stride;
      const box = mapBoxToSource(centerX, centerY, width, height, scale, padX, padY, image.width, image.height);
      if (!box) continue;
      detections.push({
        label: COCO[classIndex],
        confidence,
        ...box,
      });
    }
    const focused = focusOnObjects(detections);
    const tracked = tracker.update(focused);
    const blocked = analyzeBlockedView(image.data, image.width, { x: 0, y: 0, width: image.width, height: image.height }, tracked);
    return { available: true, detections: tracked, blocked };
  } catch {
    return { available: false, detections: [] };
  }
}

/** Browser only: the native app has no video element to read from. */
export function detectVideoFrame(_video: unknown): Promise<DetectionResult> | null {
  return null;
}

export async function warmUpDetector(): Promise<void> {}

export function localDetectorBackend(): 'webgpu' | 'wasm' | null {
  return null;
}

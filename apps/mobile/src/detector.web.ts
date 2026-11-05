import type { DetectionResult, LocalDetection } from './detector.types';

type OrtTensor = { data: Float32Array };
type OrtSession = { run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>> };
type OrtRuntime = {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  InferenceSession: { create: (url: string, options: object) => Promise<OrtSession> };
  Tensor: new (type: string, data: Float32Array, dimensions: number[]) => unknown;
};

const INPUT_SIZE = 416;
const SCORE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45;
const MAX_RESULTS = 12;
const MAX_FRAME_COVERAGE = 0.82;
const LABEL_CONFIDENCE_FLOORS: Record<string, number> = {
  // These rounded COCO classes are frequent false positives for glasses, cans,
  // lids, and other objects held close to a face.
  frisbee: 0.72,
  'sports ball': 0.66,
};
const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];

const GRID: Array<{ x: number; y: number; stride: number }> = [];
for (const stride of [8, 16, 32]) {
  const size = INPUT_SIZE / stride;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) GRID.push({ x, y, stride });
  }
}

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
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
      return ort.InferenceSession.create(assetUrl('models/yolox_nano.onnx'), {
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
    if (kept.every((existing) => existing.label !== candidate.label || iou(existing, candidate) < IOU_THRESHOLD)) {
      kept.push(candidate);
      if (kept.length === MAX_RESULTS) break;
    }
  }
  return kept;
}

function isLikelySelfPerson(item: LocalDetection) {
  return item.label === 'person' && item.width >= 0.7 && item.x <= 0.16 && item.y <= 0.28;
}

function focusOnObjects(items: LocalDetection[]) {
  const reliable = items.filter((item) => item.confidence >= (LABEL_CONFIDENCE_FLOORS[item.label] ?? SCORE_THRESHOLD));
  const nonSelf = reliable.filter((item) => !isLikelySelfPerson(item));
  // When another object is confidently visible, do not let a selfie-sized person
  // box take over the assistive UI.
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
      input[pixel] = rgba[offset + 2];
      input[plane + pixel] = rgba[offset + 1];
      input[plane * 2 + pixel] = rgba[offset];
    }
    const output = await session.run({ images: new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]) });
    const values = output.output.data as Float32Array;
    const detections: LocalDetection[] = [];
    for (let row = 0; row < GRID.length; row += 1) {
      const offset = row * 85;
      const objectness = values[offset + 4];
      if (objectness < SCORE_THRESHOLD) continue;
      let classIndex = 0;
      let classScore = 0;
      for (let index = 0; index < COCO.length; index += 1) {
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
      const box = mapBoxToSource(
        centerX, centerY, width, height, scale, padX, padY, image.naturalWidth, image.naturalHeight,
      );
      if (!box) continue;
      detections.push({
        label: COCO[classIndex], confidence, ...box,
      });
    }
    return { available: true, detections: focusOnObjects(detections) };
  } catch (error) {
    console.warn('Local YOLOX inference unavailable', error);
    return { available: false, detections: [] };
  }
}

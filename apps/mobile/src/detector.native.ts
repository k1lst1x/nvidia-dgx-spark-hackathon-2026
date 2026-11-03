import { Asset } from 'expo-asset';
import * as ImageManipulator from 'expo-image-manipulator';
import { decode } from 'jpeg-js';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import type { DetectionResult, LocalDetection } from './detector.types';

const INPUT_SIZE = 416;
const SCORE_THRESHOLD = 0.42;
const IOU_THRESHOLD = 0.45;
const MAX_RESULTS = 12;
const MODEL = require('../assets/models/yolox_nano.onnx');
const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];

let sessionPromise: Promise<InferenceSession> | null = null;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const asset = Asset.fromModule(MODEL);
      await asset.downloadAsync();
      if (!asset.localUri) throw new Error('YOLOX model asset is unavailable');
      return InferenceSession.create(asset.localUri, { executionProviders: ['xnnpack', 'cpu'] });
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

export async function detectObjects(uri: string): Promise<DetectionResult> {
  try {
    const [session, resized] = await Promise.all([
      getSession(),
      ImageManipulator.manipulateAsync(uri, [{ resize: { width: INPUT_SIZE, height: INPUT_SIZE } }], {
        compress: 0.8,
        format: ImageManipulator.SaveFormat.JPEG,
      }),
    ]);
    const bytes = new Uint8Array(await (await fetch(resized.uri)).arrayBuffer());
    const image = decode(bytes, { useTArray: true, formatAsRGBA: true });
    const plane = INPUT_SIZE * INPUT_SIZE;
    const input = new Float32Array(plane * 3);
    for (let pixel = 0; pixel < plane; pixel += 1) {
      const rgba = pixel * 4;
      input[pixel] = image.data[rgba + 2];
      input[plane + pixel] = image.data[rgba + 1];
      input[plane * 2 + pixel] = image.data[rgba];
    }
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
      const width = values[offset + 2] / INPUT_SIZE;
      const height = values[offset + 3] / INPUT_SIZE;
      detections.push({
        label: COCO[classIndex],
        confidence,
        x: Math.max(0, values[offset] / INPUT_SIZE - width / 2),
        y: Math.max(0, values[offset + 1] / INPUT_SIZE - height / 2),
        width: Math.min(1, width),
        height: Math.min(1, height),
      });
    }
    return { available: true, detections: suppressOverlaps(detections) };
  } catch {
    return { available: false, detections: [] };
  }
}

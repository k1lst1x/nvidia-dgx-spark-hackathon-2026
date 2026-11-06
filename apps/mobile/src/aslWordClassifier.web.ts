import labelsJson from '../assets/models/asl_word_labels.json';

type OrtTensor = { data: Float32Array };
type OrtSession = { run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>> };
type OrtRuntime = {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  InferenceSession: { create: (url: string, options: object) => Promise<OrtSession> };
  Tensor: new (type: string, data: Float32Array, dimensions: number[]) => unknown;
};

type LabelConfig = { classes: string[]; input_dim: number; seq_len: number };
const config = labelsJson as LabelConfig;

export type ASLWordPrediction = { label: string; confidence: number };
export type ASLWordState = {
  currentWord: string;
  currentConfidence: number;
  phrase: string;
  bufferProgress: number;
  top3: ASLWordPrediction[];
};

export type LandmarkPoint = { x: number; y: number; z?: number };

function dist2d(a: LandmarkPoint, b: LandmarkPoint) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
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
      return ort.InferenceSession.create(assetUrl('models/asl_word_transformer.onnx'), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    });
  }
  return sessionPromise;
}

function softmaxTop3(logits: Float32Array): ASLWordPrediction[] {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  let total = 0;
  const probabilities = Array.from(logits, (value) => {
    const probability = Math.exp(value - max);
    total += probability;
    return probability;
  });
  return probabilities
    .map((value, index) => ({ label: config.classes[index], confidence: value / total }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);
}

/**
 * High-speed kinematic gesture recognizer for essential ASL conversational signs
 * (hello, yes, no, please, thankyou, water).
 * Analyzes multi-frame hand geometry, finger curls, and kinetic trajectories.
 */
class KinematicGestureRecognizer {
  private history: {
    wrist: LandmarkPoint;
    indexTip: LandmarkPoint;
    indexMcp: LandmarkPoint;
    middleTip: LandmarkPoint;
    thumbTip: LandmarkPoint;
    pinkyTip: LandmarkPoint;
    isFist: boolean;
    isOpen: boolean;
    isWHand: boolean;
    pinchDist: number;
    timestamp: number;
  }[] = [];

  process(primaryHand: LandmarkPoint[] | undefined): ASLWordPrediction | null {
    if (!primaryHand || primaryHand.length < 21) {
      if (this.history.length > 0) this.history.shift();
      return null;
    }

    const wrist = primaryHand[0];
    const thumbTip = primaryHand[4];
    const indexMcp = primaryHand[5];
    const indexPip = primaryHand[6];
    const indexTip = primaryHand[8];
    const middleTip = primaryHand[12];
    const ringTip = primaryHand[16];
    const pinkyTip = primaryHand[20];

    // Finger extensions relative to wrist / MCP
    const indexExt = dist2d(indexTip, wrist) > dist2d(indexPip, wrist) * 1.15;
    const middleExt = dist2d(middleTip, wrist) > dist2d(primaryHand[10], wrist) * 1.15;
    const ringExt = dist2d(ringTip, wrist) > dist2d(primaryHand[14], wrist) * 1.15;
    const pinkyExt = dist2d(pinkyTip, wrist) > dist2d(primaryHand[18], wrist) * 1.15;

    const isFist = !indexExt && !middleExt && !ringExt && !pinkyExt;
    const isOpen = indexExt && middleExt && ringExt && pinkyExt;
    const isWHand = indexExt && middleExt && ringExt && !pinkyExt;
    const pinchDist = (dist2d(indexTip, thumbTip) + dist2d(middleTip, thumbTip)) / 2;

    this.history.push({
      wrist,
      indexTip,
      indexMcp,
      middleTip,
      thumbTip,
      pinkyTip,
      isFist,
      isOpen,
      isWHand,
      pinchDist,
      timestamp: performance.now(),
    });

    if (this.history.length > 30) this.history.shift();
    if (this.history.length < 8) return null;

    const n = this.history.length;
    const recent = this.history[n - 1];
    const older = this.history[Math.max(0, n - 12)];

    // 1. "YES" (да) - S-fist nodding vertically (up and down)
    const fistFrames = this.history.slice(-12).filter((h) => h.isFist).length;
    if (fistFrames >= 8) {
      const yValues = this.history.slice(-16).map((h) => h.wrist.y);
      let directionChanges = 0;
      let lastDir = 0;
      for (let i = 1; i < yValues.length; i++) {
        const diff = yValues[i] - yValues[i - 1];
        if (Math.abs(diff) > 0.005) {
          const dir = Math.sign(diff);
          if (lastDir !== 0 && dir !== lastDir) directionChanges++;
          lastDir = dir;
        }
      }
      const ySpan = Math.max(...yValues) - Math.min(...yValues);
      if (directionChanges >= 2 && ySpan >= 0.025) {
        return { label: 'yes', confidence: 0.92 };
      }
    }

    // 2. "NO" (нет) - Index + Middle snap shut against thumb (rapid pinch closure)
    if (!recent.isFist && !recent.isOpen) {
      const pastPinch = Math.max(...this.history.slice(-14, -4).map((h) => h.pinchDist));
      const currentPinch = Math.min(...this.history.slice(-3).map((h) => h.pinchDist));
      if (pastPinch > 0.075 && currentPinch < 0.045 && (pastPinch - currentPinch) > 0.035) {
        return { label: 'no', confidence: 0.94 };
      }
    }

    // 3. "HELLO" (привет) - Open flat hand at head/temple level moving/waving
    if (recent.isOpen && recent.wrist.y < 0.46) {
      const xValues = this.history.slice(-16).map((h) => h.wrist.x);
      let xChanges = 0;
      let lastXDir = 0;
      for (let i = 1; i < xValues.length; i++) {
        const diff = xValues[i] - xValues[i - 1];
        if (Math.abs(diff) > 0.005) {
          const dir = Math.sign(diff);
          if (lastXDir !== 0 && dir !== lastXDir) xChanges++;
          lastXDir = dir;
        }
      }
      const xSpan = Math.max(...xValues) - Math.min(...xValues);
      if ((xChanges >= 2 && xSpan >= 0.035) || xSpan >= 0.065) {
        return { label: 'hello', confidence: 0.89 };
      }
    }

    // 4. "THANK YOU" (спасибо) - Flat hand starts at chin/mouth and moves forward/down
    if (recent.isOpen) {
      const pastHighHand = this.history.slice(-15, -6).find((h) => h.indexTip.y < 0.42);
      const dy = recent.wrist.y - (pastHighHand?.wrist.y ?? recent.wrist.y);
      if (pastHighHand && dy > 0.05) {
        return { label: 'thankyou', confidence: 0.91 };
      }
    }

    // 5. "PLEASE" (плиз) - Flat hand on chest moving in a circular path
    if (recent.isOpen && recent.wrist.y >= 0.40 && recent.wrist.y <= 0.75) {
      const points = this.history.slice(-18);
      if (points.length >= 12) {
        const avgX = points.reduce((acc, p) => acc + p.wrist.x, 0) / points.length;
        const avgY = points.reduce((acc, p) => acc + p.wrist.y, 0) / points.length;
        let totalAngle = 0;
        let lastAngle: number | null = null;
        for (const p of points) {
          const ang = Math.atan2(p.wrist.y - avgY, p.wrist.x - avgX);
          if (lastAngle !== null) {
            let diff = ang - lastAngle;
            if (diff > Math.PI) diff -= 2 * Math.PI;
            if (diff < -Math.PI) diff += 2 * Math.PI;
            totalAngle += diff;
          }
          lastAngle = ang;
        }
        const radius = Math.hypot(recent.wrist.x - avgX, recent.wrist.y - avgY);
        if (Math.abs(totalAngle) > 3.2 && radius > 0.02) {
          return { label: 'please', confidence: 0.87 };
        }
      }
    }

    // 6. "WATER" (вода) - W-hand tapping near chin level
    if (recent.isWHand && recent.indexTip.y < 0.50) {
      const yValues = this.history.slice(-14).map((h) => h.indexTip.y);
      const ySpan = Math.max(...yValues) - Math.min(...yValues);
      if (ySpan > 0.02) {
        return { label: 'water', confidence: 0.88 };
      }
    }

    return null;
  }

  clear() {
    this.history = [];
  }
}

export class ASLWordEngine {
  private frames: number[][] = [];
  private inFlight = false;
  private predictionCounter = 0;
  private candidate = '';
  private stablePredictions = 0;
  private lastCommitted = '';
  private lastCommitTime = 0;
  private handsAbsentFrames = 0;
  private phrase: string[] = [];
  private kinematic = new KinematicGestureRecognizer();
  private readonly confidenceThreshold = 0.30;
  private readonly commitCooldownMs = 1200;

  constructor(private readonly onState: (state: ASLWordState) => void) {}

  processFrame(
    features?: number[],
    rawHands?: LandmarkPoint[][],
  ) {
    if (!features) {
      this.handsAbsentFrames += 1;
      if (this.handsAbsentFrames >= 8) {
        this.frames = [];
        this.candidate = '';
        this.stablePredictions = 0;
        this.kinematic.clear();
      }
      this.emit('', 0, []);
      return;
    }

    // 1. Check kinematic motion for essential conversational gestures first
    const primaryHand = rawHands?.[0];
    const kinematicMatch = this.kinematic.process(primaryHand);
    const now = performance.now();

    if (kinematicMatch && kinematicMatch.confidence >= 0.80) {
      this.candidate = kinematicMatch.label;
      if (
        kinematicMatch.label !== this.lastCommitted ||
        now - this.lastCommitTime > this.commitCooldownMs
      ) {
        this.phrase.push(kinematicMatch.label);
        this.lastCommitted = kinematicMatch.label;
        this.lastCommitTime = now;
        this.frames = [];
        this.kinematic.clear();
      }
      this.emit(kinematicMatch.label, kinematicMatch.confidence, [kinematicMatch]);
      return;
    }

    // 2. Feed into the 250-class Transformer sequence buffer
    if (features.length !== config.input_dim) return;
    this.handsAbsentFrames = 0;
    this.frames.push(features);
    if (this.frames.length > config.seq_len) this.frames.shift();
    this.predictionCounter += 1;

    // Run inference every 3 frames once at least 12 frames have accumulated
    if (this.frames.length < 12 || this.inFlight || this.predictionCounter % 3 !== 0) {
      this.emit(this.candidate, 0, []);
      return;
    }
    void this.predict();
  }

  private async predict() {
    this.inFlight = true;
    try {
      const [ort, session] = await Promise.all([getRuntime(), getSession()]);
      const data = new Float32Array(config.seq_len * config.input_dim);
      this.frames.forEach((frame, frameIndex) => data.set(frame, frameIndex * config.input_dim));
      const output = await session.run({
        landmarks: new ort.Tensor('float32', data, [1, config.seq_len, config.input_dim]),
      });
      const top3 = softmaxTop3(output.logits.data);
      const best = top3[0];
      const now = performance.now();

      if (best.confidence >= this.confidenceThreshold) {
        if (best.label === this.candidate) {
          this.stablePredictions += 1;
        } else {
          this.candidate = best.label;
          this.stablePredictions = 1;
        }

        const requiredStable = best.confidence >= 0.45 ? 2 : 3;
        if (
          this.stablePredictions >= requiredStable &&
          (best.label !== this.lastCommitted || now - this.lastCommitTime > this.commitCooldownMs)
        ) {
          this.phrase.push(best.label);
          this.lastCommitted = best.label;
          this.lastCommitTime = now;
          this.stablePredictions = 0;
        }
      } else {
        // Retain candidate for UI responsiveness if recent
        if (this.stablePredictions > 0) this.stablePredictions--;
        if (this.stablePredictions === 0) this.candidate = '';
      }
      this.emit(this.candidate || best.label, best.confidence, top3);
    } catch (error) {
      console.warn('Offline ASL word model inference warning', error);
    } finally {
      this.inFlight = false;
    }
  }

  clear() {
    this.frames = [];
    this.phrase = [];
    this.candidate = '';
    this.lastCommitted = '';
    this.stablePredictions = 0;
    this.kinematic.clear();
    this.emit('', 0, []);
  }

  private emit(word: string, confidence: number, top3: ASLWordPrediction[]) {
    this.onState({
      currentWord: word,
      currentConfidence: confidence,
      phrase: this.phrase.join(' '),
      bufferProgress: Math.min(1, this.frames.length / config.seq_len),
      top3,
    });
  }
}

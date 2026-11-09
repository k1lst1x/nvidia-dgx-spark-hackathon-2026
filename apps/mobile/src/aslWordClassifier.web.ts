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
    handSpan: number;
    scale: number;
    isFist: boolean;
    isOpen: boolean;
    isWHand: boolean;
    isPoint: boolean;
    isPeace: boolean;
    isFive: boolean;
    isThumbsUp: boolean;
    isThumbsDown: boolean;
    isOK: boolean;
    isILY: boolean;
    isPhone: boolean;
    pinchDist: number;
    timestamp: number;
  }[] = [];

  process(
    primaryHand: LandmarkPoint[] | undefined,
    secondaryHand?: LandmarkPoint[],
    face?: LandmarkPoint[],
    pose?: LandmarkPoint[],
  ): ASLWordPrediction | null {
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

    // Calculate dynamic hand span to normalize scale (invariant to distance from camera)
    const handSpan = Math.max(0.06, dist2d(wrist, indexMcp));
    const scale = handSpan / 0.12;

    // Body anchors for relative scale & position
    const nose = face?.[1] || pose?.[0];
    const chin = face?.[152] || (nose ? { x: nose.x, y: nose.y + 0.12 * scale } : undefined);
    const shoulderY = pose && pose.length > 12 ? (pose[11].y + pose[12].y) / 2 : undefined;

    // Invariant finger extensions relative to MCP and wrist
    const indexMcpDist = dist2d(indexTip, indexMcp);
    const middleMcpDist = dist2d(middleTip, primaryHand[9]);
    const ringMcpDist = dist2d(ringTip, primaryHand[13]);
    const pinkyMcpDist = dist2d(pinkyTip, primaryHand[17]);

    const indexExt = indexMcpDist > 0.07 * scale && dist2d(indexTip, wrist) > dist2d(indexPip, wrist) * 1.08;
    const middleExt = middleMcpDist > 0.07 * scale && dist2d(middleTip, wrist) > dist2d(primaryHand[10], wrist) * 1.08;
    const ringExt = ringMcpDist > 0.06 * scale && dist2d(ringTip, wrist) > dist2d(primaryHand[14], wrist) * 1.08;
    const pinkyExt = pinkyMcpDist > 0.05 * scale && dist2d(pinkyTip, wrist) > dist2d(primaryHand[18], wrist) * 1.08;
    const thumbExt = dist2d(thumbTip, primaryHand[2]) > 0.045 * scale && dist2d(thumbTip, indexMcp) > 0.045 * scale;

    const isFist = !indexExt && !middleExt && !ringExt && !pinkyExt;
    const isOpen = indexExt && middleExt && ringExt && pinkyExt;
    const isWHand = indexExt && middleExt && ringExt && !pinkyExt;
    const isPoint = indexExt && !middleExt && !ringExt && !pinkyExt && !thumbExt;
    const isPeace = indexExt && middleExt && !ringExt && !pinkyExt && !thumbExt;
    const isFive = thumbExt && indexExt && middleExt && ringExt && pinkyExt;
    const isThumbsUp = thumbExt && !indexExt && !middleExt && !ringExt && !pinkyExt && thumbTip.y < indexMcp.y - 0.03 * scale;
    const isThumbsDown = thumbExt && !indexExt && !middleExt && !ringExt && !pinkyExt && thumbTip.y > wrist.y + 0.02 * scale;
    const isPinchThumbIndex = dist2d(thumbTip, indexTip) < 0.045 * scale;
    const isOK = isPinchThumbIndex && middleExt && ringExt && pinkyExt;
    const isILY = thumbExt && indexExt && pinkyExt && !middleExt && !ringExt;
    const isPhone = thumbExt && pinkyExt && !indexExt && !middleExt && !ringExt;
    const pinchDist = (dist2d(indexTip, thumbTip) + dist2d(middleTip, thumbTip)) / 2;

    // Secondary hand relations (if both hands visible in frame)
    const hasSecHand = Boolean(secondaryHand && secondaryHand.length >= 21);
    const secWrist = hasSecHand ? secondaryHand![0] : undefined;
    const secIndexTip = hasSecHand ? secondaryHand![8] : undefined;
    const interWristDist = secWrist ? dist2d(wrist, secWrist) : 999;
    const interTipDist = secIndexTip ? dist2d(indexTip, secIndexTip) : 999;

    this.history.push({
      wrist,
      indexTip,
      indexMcp,
      middleTip,
      thumbTip,
      pinkyTip,
      handSpan,
      scale,
      isFist,
      isOpen,
      isWHand,
      isPoint,
      isPeace,
      isFive,
      isThumbsUp,
      isThumbsDown,
      isOK,
      isILY,
      isPhone,
      pinchDist,
      timestamp: performance.now(),
    });

    if (this.history.length > 30) this.history.shift();
    if (this.history.length < 6) return null;

    const n = this.history.length;
    const recent = this.history[n - 1];

    // 1. "YES" (да) - S-fist nodding vertically OR Thumbs-Up held firmly
    const thumbsUpFrames = this.history.slice(-8).filter((h) => h.isThumbsUp).length;
    if (thumbsUpFrames >= 4) {
      return { label: 'yes', confidence: 0.95 };
    }
    const fistFrames = this.history.slice(-12).filter((h) => h.isFist).length;
    if (fistFrames >= 6) {
      const yValues = this.history.slice(-16).map((h) => h.wrist.y);
      let directionChanges = 0;
      let lastDir = 0;
      for (let i = 1; i < yValues.length; i++) {
        const diff = yValues[i] - yValues[i - 1];
        if (Math.abs(diff) > 0.004 * scale) {
          const dir = Math.sign(diff);
          if (lastDir !== 0 && dir !== lastDir) directionChanges++;
          lastDir = dir;
        }
      }
      const ySpan = Math.max(...yValues) - Math.min(...yValues);
      if (directionChanges >= 2 && ySpan >= 0.018 * scale) {
        return { label: 'yes', confidence: 0.94 };
      }
    }

    // 2. "NO" (нет) - Index + Middle snap shut against thumb OR Thumbs-Down held OR index wagging
    const thumbsDownFrames = this.history.slice(-8).filter((h) => h.isThumbsDown).length;
    if (thumbsDownFrames >= 4) {
      return { label: 'no', confidence: 0.95 };
    }
    if (!recent.isFist && !recent.isOpen) {
      const pastPinch = Math.max(...this.history.slice(-14, -4).map((h) => h.pinchDist));
      const currentPinch = Math.min(...this.history.slice(-3).map((h) => h.pinchDist));
      if (pastPinch > 0.06 * scale && currentPinch < 0.045 * scale && (pastPinch - currentPinch) > 0.024 * scale) {
        return { label: 'no', confidence: 0.95 };
      }
    }
    // Index finger wagging left and right (universal "no-no-no")
    if (recent.isPoint) {
      const xValues = this.history.slice(-12).map((h) => h.indexTip.x);
      let xChanges = 0;
      let lastXDir = 0;
      for (let i = 1; i < xValues.length; i++) {
        const diff = xValues[i] - xValues[i - 1];
        if (Math.abs(diff) > 0.004 * scale) {
          const dir = Math.sign(diff);
          if (lastXDir !== 0 && dir !== lastXDir) xChanges++;
          lastXDir = dir;
        }
      }
      const xSpan = Math.max(...xValues) - Math.min(...xValues);
      if (xChanges >= 2 && xSpan > 0.02 * scale) {
        return { label: 'no', confidence: 0.93 };
      }
    }

    // 3. "PEACE" / "VICTORY" - V-sign held stably
    const peaceFrames = this.history.slice(-8).filter((h) => h.isPeace).length;
    if (peaceFrames >= 4) {
      return { label: 'peace', confidence: 0.95 };
    }

    // 4. "OKAY" / "OK" - Thumb and index forming circle with other 3 fingers extended
    const okFrames = this.history.slice(-8).filter((h) => h.isOK).length;
    if (okFrames >= 4) {
      return { label: 'okay', confidence: 0.95 };
    }

    // 5. "I LOVE YOU" (ILY sign) - Thumb, index, pinky extended
    const ilyFrames = this.history.slice(-8).filter((h) => h.isILY).length;
    if (ilyFrames >= 4) {
      return { label: 'ily', confidence: 0.96 };
    }

    // 6. "CALL" / "PHONE" - Y-hand held near ear or chin
    const isNearEarOrChin = chin ? dist2d(recent.thumbTip, chin) < 0.28 * scale : recent.wrist.y < 0.52;
    if (recent.isPhone && isNearEarOrChin) {
      return { label: 'call', confidence: 0.94 };
    }

    // 7. "HELP" (помощь) - Thumbs-up fist rising, or two-handed fist lifting from palm
    const isThumbsUpFist = isFist && thumbTip.y < indexMcp.y - 0.03 * scale;
    if (isThumbsUpFist) {
      const yValues = this.history.slice(-12).map((h) => h.wrist.y);
      const dy = yValues[yValues.length - 1] - yValues[0];
      if (dy < -0.028 * scale || (hasSecHand && interWristDist < 0.22 * scale && dy < -0.015 * scale)) {
        return { label: 'help', confidence: 0.95 };
      }
    }

    // 7. "STOP" (стоп) - Two-handed ASL chop onto palm, OR intentional rapid forward palm push
    if (recent.isOpen) {
      // A. Two-handed ASL chop: Dominant flat hand striking downward toward non-dominant palm
      if (hasSecHand && interWristDist < 0.18 * scale) {
        const pastHighDominant = this.history.slice(-12, -4).map((h) => h.wrist.y);
        const minPastY = Math.min(...pastHighDominant);
        const dy = recent.wrist.y - minPastY;
        if (dy > 0.035 * scale) {
          return { label: 'stop', confidence: 0.95 };
        }
      }
      // B. Single-handed intentional forward push: hand span expanding by >25% rapidly toward camera
      const pastSpans = this.history.slice(-12, -5).map((h) => h.handSpan);
      if (pastSpans.length >= 4) {
        const minPastSpan = Math.min(...pastSpans);
        if (minPastSpan > 0 && (recent.handSpan / minPastSpan) > 1.25) {
          return { label: 'stop', confidence: 0.92 };
        }
      }
      // NOTE: A stationary resting open hand is NEUTRAL (readiness pose), NEVER "stop"!
    }

    // 9. "HELLO" (привет) - Open flat hand at head/temple level moving/waving
    const isAtHeadLevel = nose ? recent.wrist.y <= nose.y + 0.14 * scale : recent.wrist.y < 0.48;
    if (recent.isOpen && isAtHeadLevel) {
      const xValues = this.history.slice(-16).map((h) => h.wrist.x);
      let xChanges = 0;
      let lastXDir = 0;
      for (let i = 1; i < xValues.length; i++) {
        const diff = xValues[i] - xValues[i - 1];
        if (Math.abs(diff) > 0.004 * scale) {
          const dir = Math.sign(diff);
          if (lastXDir !== 0 && dir !== lastXDir) xChanges++;
          lastXDir = dir;
        }
      }
      const xSpan = Math.max(...xValues) - Math.min(...xValues);
      if ((xChanges >= 2 && xSpan >= 0.028 * scale) || xSpan >= 0.05 * scale) {
        return { label: 'hello', confidence: 0.92 };
      }
    }

    // 10. "THANK YOU" (спасибо) - Flat hand starts at chin/mouth and moves forward/down
    if (recent.isOpen) {
      const isNearChin = (p: LandmarkPoint) => chin ? dist2d(p, chin) < 0.22 * scale : p.y < 0.45;
      const pastHighHand = this.history.slice(-15, -5).find((h) => isNearChin(h.indexTip) || isNearChin(h.wrist));
      const dy = recent.wrist.y - (pastHighHand?.wrist.y ?? recent.wrist.y);
      if (pastHighHand && dy > 0.035 * scale) {
        return { label: 'thankyou', confidence: 0.93 };
      }
    }

    // 11. "PLEASE" (пожалуйста) - Flat hand on chest moving in a circular path
    const isAtChestLevel = shoulderY
      ? recent.wrist.y >= shoulderY - 0.05 && recent.wrist.y <= shoulderY + 0.32
      : recent.wrist.y >= 0.38 && recent.wrist.y <= 0.76;

    if (recent.isOpen && isAtChestLevel) {
      const points = this.history.slice(-18);
      if (points.length >= 10) {
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
        if (Math.abs(totalAngle) > 2.6 && radius > 0.012 * scale) {
          return { label: 'please', confidence: 0.92 };
        }
      }
    }

    // 12. "SORRY" (извините) - Closed fist on chest rubbing in a circular path
    if (isFist && isAtChestLevel) {
      const points = this.history.slice(-18);
      if (points.length >= 10) {
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
        if (Math.abs(totalAngle) > 2.4 && radius > 0.012 * scale) {
          return { label: 'sorry', confidence: 0.93 };
        }
      }
    }

    // 13. "WATER" (вода) - W-hand tapping near chin level
    const isWaterNearMouth = chin ? dist2d(recent.indexTip, chin) < 0.22 * scale : recent.indexTip.y < 0.50;
    if (recent.isWHand && isWaterNearMouth) {
      const yValues = this.history.slice(-14).map((h) => h.indexTip.y);
      const ySpan = Math.max(...yValues) - Math.min(...yValues);
      if (ySpan > 0.015 * scale) {
        return { label: 'water', confidence: 0.93 };
      }
    }

    // 14. "FOOD" / "EAT" (еда) - Flattened O/pinch tapping near mouth/chin
    const isNearMouth = chin ? dist2d(recent.indexTip, chin) < 0.18 * scale : recent.indexTip.y < 0.44;
    if (recent.pinchDist < 0.05 * scale && isNearMouth) {
      const yValues = this.history.slice(-12).map((h) => h.indexTip.y);
      const ySpan = Math.max(...yValues) - Math.min(...yValues);
      if (ySpan > 0.012 * scale) {
        return { label: 'food', confidence: 0.93 };
      }
    }

    // 15. "DRINK" (пить) - C-hand tilting up near chin/mouth
    const isCupTiltNearMouth = chin ? dist2d(recent.wrist, chin) < 0.26 * scale : recent.wrist.y < 0.52;
    if (!isFist && !isOpen && thumbTip.y < indexTip.y && isCupTiltNearMouth) {
      return { label: 'drink', confidence: 0.91 };
    }

    // 16. "MORE" (ещё) - Both hands pinched, fingertips tapping together
    if (hasSecHand && recent.pinchDist < 0.06 * scale && interTipDist < 0.09 * scale) {
      return { label: 'more', confidence: 0.93 };
    }

    // 17. "DONE" / "FINISH" (всё / готово) - Both hands open flipping outward at chest level
    if (hasSecHand && recent.isOpen && isAtChestLevel && interWristDist < 0.35 * scale) {
      const xDominant = this.history.slice(-10).map((h) => h.wrist.x);
      const dx = xDominant[xDominant.length - 1] - xDominant[0];
      if (Math.abs(dx) > 0.03 * scale) {
        return { label: 'done', confidence: 0.92 };
      }
    }

    // 18. "PAIN" (боль) - Both index fingers pointing at each other in jabs
    if (hasSecHand && indexExt && !middleExt && !ringExt && interTipDist < 0.15 * scale) {
      return { label: 'pain', confidence: 0.94 };
    }

    // 19. "SAFE" (безопасно) - Both fists crossing and separating
    if (hasSecHand && isFist && interWristDist < 0.22 * scale) {
      return { label: 'safe', confidence: 0.91 };
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

  private lastHandPos?: LandmarkPoint;

  processFrame(
    features?: number[],
    rawHands?: LandmarkPoint[][],
    face?: LandmarkPoint[],
    pose?: LandmarkPoint[],
  ) {
    if (!features) {
      this.handsAbsentFrames += 1;
      if (this.handsAbsentFrames >= 8) {
        this.frames = [];
        this.candidate = '';
        this.stablePredictions = 0;
        this.kinematic.clear();
        this.lastHandPos = undefined;
      }
      this.emit('', 0, []);
      return;
    }

    // 1. Check kinematic motion for essential conversational gestures first
    const primaryHand = rawHands?.[0];
    const secondaryHand = rawHands?.[1];
    const kinematicMatch = this.kinematic.process(primaryHand, secondaryHand, face, pose);
    const now = performance.now();

    if (kinematicMatch && kinematicMatch.confidence >= 0.80) {
      this.candidate = kinematicMatch.label;
      if (
        kinematicMatch.label !== this.lastCommitted ||
        now - this.lastCommitTime > this.commitCooldownMs
      ) {
        this.lastCommitted = kinematicMatch.label;
        this.lastCommitTime = now;
        this.frames = [];
      }
      this.emit(kinematicMatch.label, kinematicMatch.confidence, [kinematicMatch]);
      return;
    }

    // When kinematic gesture finishes and hand returns to neutral, clear candidate smoothly
    if (!kinematicMatch && this.candidate && this.frames.length < 6) {
      this.candidate = '';
    }

    // 2. Motion Energy Gate (Sign Activity Detection)
    // Avoid running Transformer inference when hand is idle or resting on a table
    if (primaryHand && primaryHand.length > 0) {
      const currentWrist = primaryHand[0];
      const motion = this.lastHandPos ? dist2d(currentWrist, this.lastHandPos) : 0.05;
      this.lastHandPos = currentWrist;

      // When nearly static, slow down sequence filling to prevent false positives
      if (motion < 0.003 && this.frames.length >= 16) {
        this.emit(this.candidate, 0, []);
        return;
      }
    }

    // 3. Feed into the 250-class Transformer sequence buffer
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
          // Do not compose a transcript from isolated classifier guesses.
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

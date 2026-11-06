/** Compact, locally downloaded training samples for the causal CTC recognizer. */

export const TEMPORAL_FEATURE_COUNT = 281;

export type TrainingAnnotation = {
  id: string;
  signerId: string;
  glosses: string[];
  text?: string;
  language?: 'ASL';
};

export type LandmarkTrainingSample = TrainingAnnotation & {
  language: 'ASL';
  fps: number;
  frames: number[][];
};

/**
 * Samples an already-fused landmark stream without recording camera video.
 * An annotator must supply verified glosses when stopping the capture; this
 * class intentionally never fabricates labels from speech or text.
 */
export class SignRecorder {
  private frames: number[][] = [];
  private lastTimestamp = -Infinity;
  private active = false;

  constructor(private readonly fps = 20) {
    if (!Number.isFinite(fps) || fps <= 0) throw new Error('Recorder FPS must be positive');
  }

  get recording() {
    return this.active;
  }

  start() {
    this.frames = [];
    this.lastTimestamp = -Infinity;
    this.active = true;
  }

  addFrame(features: number[], timestamp: number) {
    if (!this.active || timestamp - this.lastTimestamp < 1000 / this.fps) return;
    if (features.length !== TEMPORAL_FEATURE_COUNT) {
      throw new Error(`Expected ${TEMPORAL_FEATURE_COUNT} temporal features, got ${features.length}`);
    }
    if (!features.every(Number.isFinite)) throw new Error('Landmark features must be finite numbers');
    this.frames.push([...features]);
    this.lastTimestamp = timestamp;
  }

  stop(annotation: TrainingAnnotation): LandmarkTrainingSample {
    this.active = false;
    if (!annotation.id.trim() || !annotation.signerId.trim()) throw new Error('A sample ID and signer ID are required');
    const glosses = annotation.glosses.map((gloss) => gloss.trim().toUpperCase()).filter(Boolean);
    if (!glosses.length) throw new Error('Verified gloss labels are required');
    if (!this.frames.length) throw new Error('No landmark frames were captured');
    const sample: LandmarkTrainingSample = {
      id: annotation.id.trim(),
      signerId: annotation.signerId.trim(),
      language: 'ASL',
      glosses,
      text: annotation.text?.trim(),
      fps: this.fps,
      frames: this.frames,
    };
    this.frames = [];
    return sample;
  }
}

export function downloadTrainingSample(sample: LandmarkTrainingSample) {
  const blob = new Blob([JSON.stringify(sample)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${sample.id}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

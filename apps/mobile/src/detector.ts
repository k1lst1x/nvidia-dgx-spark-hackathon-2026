import type { DetectionResult } from './detector.types';

// TypeScript/default-platform fallback. Metro selects detector.native.ts on iOS/Android
// and detector.web.ts in browsers.
export async function detectObjects(_uri: string): Promise<DetectionResult> {
  return { available: false, detections: [] };
}

/** Browser only: detect from the live camera video. Null when not supported. */
export function detectVideoFrame(_video: unknown): Promise<DetectionResult> | null {
  return null;
}

export async function warmUpDetector(): Promise<void> {}

export function localDetectorBackend(): 'webgpu' | 'wasm' | null {
  return null;
}

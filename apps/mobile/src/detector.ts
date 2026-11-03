import type { DetectionResult } from './detector.types';

// TypeScript/default-platform fallback. Metro selects detector.native.ts on iOS/Android
// and detector.web.ts in browsers.
export async function detectObjects(_uri: string): Promise<DetectionResult> {
  return { available: false, detections: [] };
}

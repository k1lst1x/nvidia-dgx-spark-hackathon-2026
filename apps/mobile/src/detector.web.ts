import type { DetectionResult } from './detector.types';

export async function detectObjects(_uri: string): Promise<DetectionResult> {
  return { available: false, detections: [] };
}

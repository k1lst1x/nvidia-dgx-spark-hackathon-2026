export type LocalDetection = {
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  distance_m?: number;
  direction?: 'left' | 'ahead' | 'right';
  id?: number;
  is_approaching?: boolean;
  velocity_mps?: number;
};

export type DetectionResult = {
  available: boolean;
  detections: LocalDetection[];
  /** Directions where the frame is featureless or dark (see blockedView.ts). */
  blocked?: import('./blockedView').BlockedView | null;
};

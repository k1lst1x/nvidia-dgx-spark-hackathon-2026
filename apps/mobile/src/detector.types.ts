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
};

export type DetectionResult = {
  available: boolean;
  detections: LocalDetection[];
};

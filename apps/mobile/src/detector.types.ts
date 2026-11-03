export type LocalDetection = {
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type DetectionResult = {
  available: boolean;
  detections: LocalDetection[];
};

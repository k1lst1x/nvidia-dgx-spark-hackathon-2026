import type { VisionObject, VisionPerformance } from '../src/vision';

type Props = {
  detections: Array<Pick<VisionObject, 'label' | 'confidence' | 'x' | 'y' | 'width' | 'height' | 'distance_m' | 'track_id' | 'spatial'>>;
  performance: VisionPerformance;
};

export default function LiveSceneMap(_props: Props) {
  return null;
}

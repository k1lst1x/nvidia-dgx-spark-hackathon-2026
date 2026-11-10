import type { BlockedFrame, ObstacleFrame, VisionObject, VisionPerformance } from '../src/vision';

type Props = {
  detections: Array<Pick<VisionObject, 'label' | 'confidence' | 'x' | 'y' | 'width' | 'height' | 'distance_m' | 'track_id' | 'spatial'>>;
  obstacles?: ObstacleFrame | null;
  blocked?: BlockedFrame | null;
  performance: VisionPerformance;
};

export default function LiveSceneMap(_props: Props) {
  return null;
}

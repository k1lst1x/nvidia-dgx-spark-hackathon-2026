type SignContextState = {
  frameCount: number;
  maxFrames: number;
  handCount: number;
  ready: boolean;
  error?: string;
};

type HandTrackerProps = {
  onContextChange?: (state: SignContextState) => void;
  onContextCaptured?: (sequence: number[][]) => void;
};

export default function HandTracker(_props: HandTrackerProps) {
  return null;
}

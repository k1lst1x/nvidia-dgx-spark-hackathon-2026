// TypeScript resolves this shared module while Metro selects HandTracker.web.tsx
// or HandTracker.native.tsx at runtime.
export type SignContextState = {
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

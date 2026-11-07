import type { ASLSessionState } from '../src/aslClassifier';
import type { ASLWordState } from '../src/aslWordClassifier.web';

export type SignContextState = {
  frameCount: number;
  maxFrames: number;
  handCount: number;
  ready: boolean;
  error?: string;
  aslState?: ASLSessionState;
  faceTracked?: boolean;
  poseTracked?: boolean;
  wordState?: ASLWordState;
};

export type HandTrackerProps = {
  onContextChange?: (state: SignContextState) => void;
  onContextCaptured?: (sequence: number[][], frames: string[]) => void;
  onAslStateChange?: (state: ASLSessionState) => void;
  aslEngineRef?: React.MutableRefObject<{
    backspace: () => string;
    addSpace: () => string;
    clearText: () => void;
  } | null>;
};

export default function HandTracker(_props: HandTrackerProps) {
  return null;
}

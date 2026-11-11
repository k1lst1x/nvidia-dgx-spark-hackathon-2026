/**
 * The 281-value sign feature frame, schema "visionmate-281-v1".
 * Specification: docs/sign/FEATURE_SCHEMA_281.md.
 *
 * This is the only place the layout is defined. Any other producer (for
 * example a dataset conversion script) must reproduce the test vector in
 * tests/signFeatures.test.cjs exactly.
 */

export const SIGN_FEATURE_SCHEMA = 'visionmate-281-v1';
export const SIGN_FEATURE_COUNT = 281;

export type FeaturePoint = { x: number; y: number; z?: number };

/**
 * The 52 MediaPipe face blendshapes, in the fixed order used by the schema
 * (alphabetical). Looked up by name, so the order MediaPipe returns them in,
 * and the browser's locale, cannot change the vector.
 */
export const BLENDSHAPE_ORDER = [
  '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
  'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
  'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
  'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
  'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
  'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
] as const;

const BLENDSHAPE_INDEX = new Map<string, number>(BLENDSHAPE_ORDER.map((name, index) => [name, index]));

export const SIGN_FEATURE_OFFSETS = { leftHand: 0, rightHand: 63, pose: 126, blendshapes: 225, presence: 277 } as const;

function writePoints(target: number[], offset: number, points: FeaturePoint[] | null | undefined, count: number) {
  if (!points) return;
  for (let index = 0; index < count; index += 1) {
    const point = points[index];
    if (!point) continue;
    target[offset + index * 3] = point.x;
    target[offset + index * 3 + 1] = point.y;
    target[offset + index * 3 + 2] = point.z ?? 0;
  }
}

export function buildSignFeatureFrame(input: {
  leftHand?: FeaturePoint[] | null;
  rightHand?: FeaturePoint[] | null;
  pose?: FeaturePoint[] | null;
  facePresent: boolean;
  blendshapes?: Array<{ categoryName: string; score: number }> | null;
}): number[] {
  const frame = Array<number>(SIGN_FEATURE_COUNT).fill(0);
  writePoints(frame, SIGN_FEATURE_OFFSETS.leftHand, input.leftHand, 21);
  writePoints(frame, SIGN_FEATURE_OFFSETS.rightHand, input.rightHand, 21);
  writePoints(frame, SIGN_FEATURE_OFFSETS.pose, input.pose, 33);
  input.blendshapes?.forEach((shape) => {
    const index = BLENDSHAPE_INDEX.get(shape.categoryName);
    if (index != null && Number.isFinite(shape.score)) frame[SIGN_FEATURE_OFFSETS.blendshapes + index] = shape.score;
  });
  frame[SIGN_FEATURE_OFFSETS.presence] = input.leftHand ? 1 : 0;
  frame[SIGN_FEATURE_OFFSETS.presence + 1] = input.rightHand ? 1 : 0;
  frame[SIGN_FEATURE_OFFSETS.presence + 2] = input.facePresent ? 1 : 0;
  frame[SIGN_FEATURE_OFFSETS.presence + 3] = input.pose ? 1 : 0;
  return frame;
}

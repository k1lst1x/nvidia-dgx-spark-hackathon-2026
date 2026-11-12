/**
 * Canonical 21-point MediaPipe hand landmark poses and animated keyframes
 * for English-to-ASL sign translation and fingerspelling.
 *
 * Each hand pose contains 21 landmarks [x, y, z] normalized to [0, 1] frame space:
 * 0: Wrist
 * 1..4: Thumb (CMC, MCP, IP, Tip)
 * 5..8: Index (MCP, PIP, DIP, Tip)
 * 9..12: Middle (MCP, PIP, DIP, Tip)
 * 13..16: Ring (MCP, PIP, DIP, Tip)
 * 17..20: Pinky (MCP, PIP, DIP, Tip)
 */

export type LandmarkPoint = { x: number; y: number; z: number };
export type HandPose = LandmarkPoint[];

export type SignKeyframe = {
  /** Hand landmarks for this frame (21 points) */
  landmarks: HandPose;
  /** Secondary non-dominant hand landmarks if two-handed sign */
  secondaryLandmarks?: HandPose;
  /** Translation offset in 3D camera space for moving signs (dx, dy, dz) */
  motionOffset?: { x: number; y: number; z: number };
  /** Duration of this keyframe in milliseconds */
  durationMs: number;
};

export type SignDefinition = {
  gloss: string;
  englishLabel: string;
  type: 'word' | 'letter' | 'number';
  description: string;
  category?: 'conversation' | 'emergency' | 'navigation' | 'fingerspelling' | 'numbers';
  keyframes: SignKeyframe[];
};

export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm Base knuckles
  [5, 9], [9, 13], [13, 17],
];

// Base wrist and knuckle coordinates for standard hand rig
const W_X = 0.50, W_Y = 0.78;
const MCP_5X = 0.43, MCP_5Y = 0.54; // Index MCP
const MCP_9X = 0.49, MCP_9Y = 0.51; // Middle MCP
const MCP_13X = 0.55, MCP_13Y = 0.53; // Ring MCP
const MCP_17X = 0.60, MCP_17Y = 0.57; // Pinky MCP

/** Helper to generate a standardized hand shape based on finger extension states */
export function buildHandPose(options: {
  thumb: 'open' | 'folded' | 'fist' | 'up' | 'out' | 'touch_index' | 'touch_middle' | 'touch_ring' | 'touch_pinky';
  index: 'extended' | 'curled' | 'hooked' | 'horizontal' | 'folded';
  middle: 'extended' | 'curled' | 'horizontal' | 'folded' | 'crossed';
  ring: 'extended' | 'curled' | 'folded';
  pinky: 'extended' | 'curled' | 'folded';
  wristOffset?: { x: number; y: number; z: number };
}): HandPose {
  const ox = options.wristOffset?.x ?? 0;
  const oy = options.wristOffset?.y ?? 0;
  const oz = options.wristOffset?.z ?? 0;

  const wrist: LandmarkPoint = { x: W_X + ox, y: W_Y + oy, z: oz };

  // Thumb
  const thumbCmc: LandmarkPoint = { x: 0.46 + ox, y: 0.72 + oy, z: -0.01 + oz };
  let thumbMcp: LandmarkPoint, thumbIp: LandmarkPoint, thumbTip: LandmarkPoint;
  if (options.thumb === 'out' || options.thumb === 'open') {
    thumbMcp = { x: 0.38 + ox, y: 0.68 + oy, z: -0.02 + oz };
    thumbIp = { x: 0.31 + ox, y: 0.62 + oy, z: -0.03 + oz };
    thumbTip = { x: 0.25 + ox, y: 0.58 + oy, z: -0.04 + oz };
  } else if (options.thumb === 'up') {
    thumbMcp = { x: 0.40 + ox, y: 0.65 + oy, z: -0.02 + oz };
    thumbIp = { x: 0.36 + ox, y: 0.56 + oy, z: -0.03 + oz };
    thumbTip = { x: 0.34 + ox, y: 0.47 + oy, z: -0.04 + oz };
  } else if (options.thumb === 'touch_index') {
    thumbMcp = { x: 0.42 + ox, y: 0.66 + oy, z: -0.02 + oz };
    thumbIp = { x: 0.41 + ox, y: 0.56 + oy, z: 0.02 + oz };
    thumbTip = { x: 0.43 + ox, y: 0.47 + oy, z: 0.04 + oz };
  } else if (options.thumb === 'touch_middle') {
    thumbMcp = { x: 0.43 + ox, y: 0.66 + oy, z: -0.02 + oz };
    thumbIp = { x: 0.45 + ox, y: 0.56 + oy, z: 0.02 + oz };
    thumbTip = { x: 0.49 + ox, y: 0.48 + oy, z: 0.04 + oz };
  } else if (options.thumb === 'touch_ring') {
    thumbMcp = { x: 0.45 + ox, y: 0.66 + oy, z: -0.02 + oz };
    thumbIp = { x: 0.49 + ox, y: 0.57 + oy, z: 0.02 + oz };
    thumbTip = { x: 0.54 + ox, y: 0.50 + oy, z: 0.04 + oz };
  } else if (options.thumb === 'touch_pinky') {
    thumbMcp = { x: 0.46 + ox, y: 0.66 + oy, z: -0.02 + oz };
    thumbIp = { x: 0.52 + ox, y: 0.58 + oy, z: 0.02 + oz };
    thumbTip = { x: 0.58 + ox, y: 0.52 + oy, z: 0.04 + oz };
  } else if (options.thumb === 'fist') {
    thumbMcp = { x: 0.43 + ox, y: 0.65 + oy, z: 0.02 + oz };
    thumbIp = { x: 0.46 + ox, y: 0.58 + oy, z: 0.04 + oz };
    thumbTip = { x: 0.51 + ox, y: 0.59 + oy, z: 0.04 + oz };
  } else {
    // Folded
    thumbMcp = { x: 0.44 + ox, y: 0.66 + oy, z: 0.01 + oz };
    thumbIp = { x: 0.47 + ox, y: 0.60 + oy, z: 0.02 + oz };
    thumbTip = { x: 0.49 + ox, y: 0.56 + oy, z: 0.02 + oz };
  }

  // Index
  const indexMcp: LandmarkPoint = { x: MCP_5X + ox, y: MCP_5Y + oy, z: oz };
  let indexPip: LandmarkPoint, indexDip: LandmarkPoint, indexTip: LandmarkPoint;
  if (options.index === 'extended') {
    indexPip = { x: 0.42 + ox, y: 0.43 + oy, z: -0.01 + oz };
    indexDip = { x: 0.41 + ox, y: 0.34 + oy, z: -0.02 + oz };
    indexTip = { x: 0.40 + ox, y: 0.25 + oy, z: -0.03 + oz };
  } else if (options.index === 'hooked') {
    indexPip = { x: 0.42 + ox, y: 0.43 + oy, z: -0.01 + oz };
    indexDip = { x: 0.44 + ox, y: 0.45 + oy, z: 0.03 + oz };
    indexTip = { x: 0.44 + ox, y: 0.51 + oy, z: 0.02 + oz };
  } else if (options.index === 'horizontal') {
    indexPip = { x: 0.36 + ox, y: 0.54 + oy, z: -0.01 + oz };
    indexDip = { x: 0.28 + ox, y: 0.54 + oy, z: -0.02 + oz };
    indexTip = { x: 0.20 + ox, y: 0.54 + oy, z: -0.03 + oz };
  } else {
    // Curled / Folded
    indexPip = { x: 0.43 + ox, y: 0.47 + oy, z: 0.02 + oz };
    indexDip = { x: 0.43 + ox, y: 0.54 + oy, z: 0.04 + oz };
    indexTip = { x: 0.44 + ox, y: 0.60 + oy, z: 0.03 + oz };
  }

  // Middle
  const middleMcp: LandmarkPoint = { x: MCP_9X + ox, y: MCP_9Y + oy, z: oz };
  let middlePip: LandmarkPoint, middleDip: LandmarkPoint, middleTip: LandmarkPoint;
  if (options.middle === 'extended') {
    middlePip = { x: 0.49 + ox, y: 0.40 + oy, z: -0.01 + oz };
    middleDip = { x: 0.49 + ox, y: 0.31 + oy, z: -0.02 + oz };
    middleTip = { x: 0.49 + ox, y: 0.22 + oy, z: -0.03 + oz };
  } else if (options.middle === 'crossed') {
    middlePip = { x: 0.47 + ox, y: 0.40 + oy, z: 0.01 + oz };
    middleDip = { x: 0.44 + ox, y: 0.32 + oy, z: 0.02 + oz };
    middleTip = { x: 0.41 + ox, y: 0.23 + oy, z: 0.03 + oz };
  } else if (options.middle === 'horizontal') {
    middlePip = { x: 0.42 + ox, y: 0.51 + oy, z: -0.01 + oz };
    middleDip = { x: 0.33 + ox, y: 0.51 + oy, z: -0.02 + oz };
    middleTip = { x: 0.24 + ox, y: 0.51 + oy, z: -0.03 + oz };
  } else {
    // Curled
    middlePip = { x: 0.49 + ox, y: 0.44 + oy, z: 0.02 + oz };
    middleDip = { x: 0.49 + ox, y: 0.52 + oy, z: 0.04 + oz };
    middleTip = { x: 0.49 + ox, y: 0.59 + oy, z: 0.03 + oz };
  }

  // Ring
  const ringMcp: LandmarkPoint = { x: MCP_13X + ox, y: MCP_13Y + oy, z: oz };
  let ringPip: LandmarkPoint, ringDip: LandmarkPoint, ringTip: LandmarkPoint;
  if (options.ring === 'extended') {
    ringPip = { x: 0.55 + ox, y: 0.42 + oy, z: -0.01 + oz };
    ringDip = { x: 0.56 + ox, y: 0.33 + oy, z: -0.02 + oz };
    ringTip = { x: 0.56 + ox, y: 0.24 + oy, z: -0.03 + oz };
  } else {
    // Curled
    ringPip = { x: 0.55 + ox, y: 0.46 + oy, z: 0.02 + oz };
    ringDip = { x: 0.55 + ox, y: 0.54 + oy, z: 0.04 + oz };
    ringTip = { x: 0.54 + ox, y: 0.60 + oy, z: 0.03 + oz };
  }

  // Pinky
  const pinkyMcp: LandmarkPoint = { x: MCP_17X + ox, y: MCP_17Y + oy, z: oz };
  let pinkyPip: LandmarkPoint, pinkyDip: LandmarkPoint, pinkyTip: LandmarkPoint;
  if (options.pinky === 'extended') {
    pinkyPip = { x: 0.62 + ox, y: 0.47 + oy, z: -0.01 + oz };
    pinkyDip = { x: 0.64 + ox, y: 0.39 + oy, z: -0.02 + oz };
    pinkyTip = { x: 0.65 + ox, y: 0.30 + oy, z: -0.03 + oz };
  } else {
    // Curled
    pinkyPip = { x: 0.60 + ox, y: 0.50 + oy, z: 0.02 + oz };
    pinkyDip = { x: 0.59 + ox, y: 0.57 + oy, z: 0.04 + oz };
    pinkyTip = { x: 0.58 + ox, y: 0.62 + oy, z: 0.03 + oz };
  }

  return [
    wrist,
    thumbCmc, thumbMcp, thumbIp, thumbTip,
    indexMcp, indexPip, indexDip, indexTip,
    middleMcp, middlePip, middleDip, middleTip,
    ringMcp, ringPip, ringDip, ringTip,
    pinkyMcp, pinkyPip, pinkyDip, pinkyTip,
  ];
}

/** Standard Neutral Resting Hand Pose */
export const REST_POSE = buildHandPose({
  thumb: 'open',
  index: 'extended',
  middle: 'extended',
  ring: 'extended',
  pinky: 'extended',
  wristOffset: { x: 0, y: 0.08, z: 0 },
});

// Canonical ASL Fingerspelling Hand Poses (A - Z)
export const LETTER_POSES: Record<string, HandPose> = {
  A: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  B: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended' }),
  C: buildHandPose({ thumb: 'open', index: 'hooked', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  D: buildHandPose({ thumb: 'touch_middle', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  E: buildHandPose({ thumb: 'folded', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  F: buildHandPose({ thumb: 'touch_index', index: 'curled', middle: 'extended', ring: 'extended', pinky: 'extended' }),
  G: buildHandPose({ thumb: 'out', index: 'horizontal', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  H: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled' }),
  I: buildHandPose({ thumb: 'folded', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'extended' }),
  J: buildHandPose({ thumb: 'folded', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'extended', wristOffset: { x: -0.05, y: 0, z: 0 } }),
  K: buildHandPose({ thumb: 'up', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled' }),
  L: buildHandPose({ thumb: 'out', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  M: buildHandPose({ thumb: 'folded', index: 'folded', middle: 'folded', ring: 'folded', pinky: 'curled' }),
  N: buildHandPose({ thumb: 'folded', index: 'folded', middle: 'folded', ring: 'curled', pinky: 'curled' }),
  O: buildHandPose({ thumb: 'touch_index', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  P: buildHandPose({ thumb: 'up', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0.08, z: 0 } }),
  Q: buildHandPose({ thumb: 'out', index: 'horizontal', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0.08, z: 0 } }),
  R: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'crossed', ring: 'curled', pinky: 'curled' }),
  S: buildHandPose({ thumb: 'fist', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  T: buildHandPose({ thumb: 'up', index: 'folded', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  U: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled' }),
  V: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled' }),
  W: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'curled' }),
  X: buildHandPose({ thumb: 'folded', index: 'hooked', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  Y: buildHandPose({ thumb: 'out', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'extended' }),
  Z: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.05, y: -0.05, z: 0 } }),
};

// Digits 0 - 9
export const DIGIT_POSES: Record<string, HandPose> = {
  0: LETTER_POSES.O,
  1: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled' }),
  2: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled' }),
  3: buildHandPose({ thumb: 'out', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled' }),
  4: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended' }),
  5: buildHandPose({ thumb: 'out', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended' }),
  6: buildHandPose({ thumb: 'touch_pinky', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'curled' }),
  7: buildHandPose({ thumb: 'touch_ring', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'extended' }),
  8: buildHandPose({ thumb: 'touch_middle', index: 'extended', middle: 'curled', ring: 'extended', pinky: 'extended' }),
  9: buildHandPose({ thumb: 'touch_index', index: 'curled', middle: 'extended', ring: 'extended', pinky: 'extended' }),
};

/**
 * Animated ASL Signs Dictionary (Whole Words) with multi-frame keyframes
 */
export const ASL_DICTIONARY: Record<string, SignDefinition> = {
  HELLO: {
    gloss: 'HELLO',
    englishLabel: 'Hello / Hi',
    type: 'word',
    category: 'conversation',
    description: 'Open flat hand near temple, saluting and waving outward to the side.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.05, y: -0.22, z: 0 } }),
        durationMs: 380,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.18, y: -0.16, z: 0 } }),
        durationMs: 440,
      },
    ],
  },

  'THANK-YOU': {
    gloss: 'THANK-YOU',
    englishLabel: 'Thank you / Thanks',
    type: 'word',
    category: 'conversation',
    description: 'Flat hand fingertips touch chin, then extend forward and downward toward the viewer.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0, y: -0.18, z: -0.05 } }),
        durationMs: 360,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0, y: 0.05, z: 0.10 } }),
        durationMs: 480,
      },
    ],
  },

  PLEASE: {
    gloss: 'PLEASE',
    englishLabel: 'Please',
    type: 'word',
    category: 'conversation',
    description: 'Open flat hand rubs clockwise circle over the center of the chest.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.06, y: -0.02, z: 0 } }),
        durationMs: 280,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.06, y: -0.08, z: 0 } }),
        durationMs: 280,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.06, y: 0.04, z: 0 } }),
        durationMs: 280,
      },
    ],
  },

  HELP: {
    gloss: 'HELP',
    englishLabel: 'Help / Assist',
    type: 'word',
    category: 'emergency',
    description: 'Dominant fist with thumb pointing up rests on flat non-dominant palm, lifting together upward.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0.06, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.15, y: 0.12, z: 0 } }),
        durationMs: 350,
      },
      {
        landmarks: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: -0.15, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.15, y: -0.08, z: 0 } }),
        durationMs: 480,
      },
    ],
  },

  YES: {
    gloss: 'YES',
    englishLabel: 'Yes / Correct',
    type: 'word',
    category: 'conversation',
    description: 'S-fist nods up and down vertically like a nodding head.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'fist', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: -0.06, z: 0 } }),
        durationMs: 220,
      },
      {
        landmarks: buildHandPose({ thumb: 'fist', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0.08, z: 0 } }),
        durationMs: 220,
      },
      {
        landmarks: buildHandPose({ thumb: 'fist', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: -0.04, z: 0 } }),
        durationMs: 220,
      },
      {
        landmarks: buildHandPose({ thumb: 'fist', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0.06, z: 0 } }),
        durationMs: 220,
      },
    ],
  },

  NO: {
    gloss: 'NO',
    englishLabel: 'No / Negative',
    type: 'word',
    category: 'conversation',
    description: 'Index and middle fingers snap down together firmly against the thumb tip.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'out', index: 'extended', middle: 'extended', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0, z: 0 } }),
        durationMs: 260,
      },
      {
        landmarks: buildHandPose({ thumb: 'touch_index', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0, y: 0.04, z: 0 } }),
        durationMs: 400,
      },
    ],
  },

  STOP: {
    gloss: 'STOP',
    englishLabel: 'Stop / Halt',
    type: 'word',
    category: 'emergency',
    description: 'Dominant hand knife-edge chops downward firmly onto the open flat palm of the non-dominant hand.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.08, y: -0.18, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.12, y: 0.08, z: 0 } }),
        durationMs: 260,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.04, y: 0.05, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.12, y: 0.08, z: 0 } }),
        durationMs: 460,
      },
    ],
  },

  'I-LOVE-YOU': {
    gloss: 'I-LOVE-YOU',
    englishLabel: 'I love you (ILY)',
    type: 'word',
    category: 'conversation',
    description: 'Thumb, index, and pinky fingers extended upright facing partner with subtle forward wave.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'out', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'extended', wristOffset: { x: 0, y: -0.08, z: 0 } }),
        durationMs: 400,
      },
      {
        landmarks: buildHandPose({ thumb: 'out', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'extended', wristOffset: { x: 0, y: -0.04, z: 0.06 } }),
        durationMs: 440,
      },
    ],
  },

  WATER: {
    gloss: 'WATER',
    englishLabel: 'Water / Drink',
    type: 'word',
    category: 'conversation',
    description: 'W-hand (index, middle, ring extended) index finger taps chin / lower lip twice.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'curled', wristOffset: { x: 0, y: -0.18, z: -0.04 } }),
        durationMs: 220,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'curled', wristOffset: { x: 0, y: -0.12, z: 0 } }),
        durationMs: 180,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'curled', wristOffset: { x: 0, y: -0.18, z: -0.04 } }),
        durationMs: 220,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'curled', wristOffset: { x: 0, y: -0.12, z: 0 } }),
        durationMs: 240,
      },
    ],
  },

  ROOM: {
    gloss: 'ROOM',
    englishLabel: 'Room / Office / Classroom',
    type: 'word',
    category: 'navigation',
    description: 'Both flat hands outline the parallel walls of a room (front-and-back, then side-to-side).',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.12, y: 0, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.12, y: 0, z: 0 } }),
        durationMs: 380,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.08, y: 0.08, z: 0.10 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.08, y: 0.08, z: -0.10 } }),
        durationMs: 440,
      },
    ],
  },

  WHERE: {
    gloss: 'WHERE',
    englishLabel: 'Where / Location',
    type: 'word',
    category: 'navigation',
    description: 'Index finger pointing upward shakes side-to-side with questioning expression.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.06, y: -0.04, z: 0 } }),
        durationMs: 240,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.06, y: -0.04, z: 0 } }),
        durationMs: 240,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.05, y: -0.04, z: 0 } }),
        durationMs: 240,
      },
    ],
  },

  FOLLOW: {
    gloss: 'FOLLOW',
    englishLabel: 'Follow me',
    type: 'word',
    category: 'navigation',
    description: 'Both hands with thumbs up; dominant hand trails right behind non-dominant hand moving forward.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.04, y: 0.06, z: -0.08 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.08, y: 0, z: 0.06 } }),
        durationMs: 380,
      },
      {
        landmarks: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.04, y: -0.08, z: 0.08 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'up', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.08, y: -0.14, z: 0.16 } }),
        durationMs: 440,
      },
    ],
  },

  WELCOME: {
    gloss: 'WELCOME',
    englishLabel: "You're welcome / Welcome",
    type: 'word',
    category: 'conversation',
    description: 'Open flat hand sweeps gently from outward inward toward the waist/center.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.18, y: -0.02, z: 0.04 } }),
        durationMs: 340,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.02, y: 0.10, z: -0.04 } }),
        durationMs: 440,
      },
    ],
  },

  PHONE: {
    gloss: 'PHONE',
    englishLabel: 'Phone / Call',
    type: 'word',
    category: 'conversation',
    description: 'Y-hand (thumb and pinky extended) held with thumb to ear and pinky toward mouth.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'out', index: 'curled', middle: 'curled', ring: 'curled', pinky: 'extended', wristOffset: { x: 0.12, y: -0.24, z: -0.02 } }),
        durationMs: 480,
      },
    ],
  },

  WAIT: {
    gloss: 'WAIT',
    englishLabel: 'Wait / Hold on',
    type: 'word',
    category: 'conversation',
    description: 'Both hands held forward with open palms facing body and wiggling fingers.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0.10, y: -0.02, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.10, y: -0.02, z: 0 } }),
        durationMs: 300,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'hooked', middle: 'curled', ring: 'curled', pinky: 'extended', wristOffset: { x: 0.10, y: 0.02, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'hooked', middle: 'curled', ring: 'curled', pinky: 'extended', wristOffset: { x: -0.10, y: 0.02, z: 0 } }),
        durationMs: 300,
      },
    ],
  },

  GOOD: {
    gloss: 'GOOD',
    englishLabel: 'Good / Well',
    type: 'word',
    category: 'conversation',
    description: 'Flat hand touches chin, then moves downward into the palm of the other hand.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0, y: -0.16, z: -0.02 } }),
        durationMs: 320,
      },
      {
        landmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: 0, y: 0.06, z: 0.08 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'open', index: 'extended', middle: 'extended', ring: 'extended', pinky: 'extended', wristOffset: { x: -0.12, y: 0.10, z: 0 } }),
        durationMs: 440,
      },
    ],
  },

  NAME: {
    gloss: 'NAME',
    englishLabel: 'Name',
    type: 'word',
    category: 'conversation',
    description: 'Two H-hands (index and middle extended) tap across each other at knuckles twice.',
    keyframes: [
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.04, y: 0, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.04, y: 0.06, z: 0 } }),
        durationMs: 250,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.04, y: 0.05, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.04, y: 0.06, z: 0 } }),
        durationMs: 220,
      },
      {
        landmarks: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled', wristOffset: { x: 0.04, y: 0, z: 0 } }),
        secondaryLandmarks: buildHandPose({ thumb: 'folded', index: 'horizontal', middle: 'horizontal', ring: 'curled', pinky: 'curled', wristOffset: { x: -0.04, y: 0.06, z: 0 } }),
        durationMs: 250,
      },
    ],
  },
};

/** Linear interpolation between two 21-point hand poses */
export function interpolatePose(a: HandPose, b: HandPose, t: number): HandPose {
  const clampedT = Math.max(0, Math.min(1, t));
  // Smooth cosine easing
  const ease = (1 - Math.cos(clampedT * Math.PI)) / 2;
  const result: HandPose = new Array(21);
  for (let i = 0; i < 21; i++) {
    result[i] = {
      x: a[i].x + (b[i].x - a[i].x) * ease,
      y: a[i].y + (b[i].y - a[i].y) * ease,
      z: a[i].z + (b[i].z - a[i].z) * ease,
    };
  }
  return result;
}

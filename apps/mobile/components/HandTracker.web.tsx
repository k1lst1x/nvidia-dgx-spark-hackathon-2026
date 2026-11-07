import { CSSProperties, useEffect, useRef } from 'react';
import { ASLGestureEngine, type ASLSessionState } from '../src/aslClassifier';
import { ASLWordEngine, type ASLWordState } from '../src/aslWordClassifier.web';
import { TEMPORAL_FEATURE_COUNT, type RawHolisticFrame, type SignRecorder } from '../src/landmarkRecorder.web';

const wasmRoot = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/wasm';
const modelPath = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const faceModelPath = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const poseModelPath = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
// Pin the task asset version. `latest` is not a supported GCS model version
// and caused the holistic path to be silently unreachable in some browsers.
const holisticModelPath = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task';
const visionModule = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/vision_bundle.mjs';
const framesPerGesture = 30;
const contextGestures = 6;
const maxContextFrames = framesPerGesture * contextGestures;
const minimumFramesForTranslation = framesPerGesture;
const snapshotCount = 3;
const snapshotIntervalFrames = minimumFramesForTranslation / snapshotCount;
const liveUpdateIntervalMs = 2000;
// Preserve the entire front-camera frame so signing can include hands, face,
// and upper body. Both layers use the same fit to keep landmarks aligned.

type Landmark = { x: number; y: number; z?: number };
type Handedness = Array<Array<{ categoryName?: string; displayName?: string; score?: number }>>;
type HandDetector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => {
    landmarks: Landmark[][];
    handednesses?: Handedness;
    handedness?: Handedness;
  };
  close: () => void;
};

type FaceDetector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => {
    faceLandmarks: Landmark[][];
    faceBlendshapes?: Array<{ categories: Array<{ categoryName: string; score: number }> }>;
  };
  close: () => void;
};

type PoseDetector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => { landmarks: Landmark[][] };
  close: () => void;
};

type HolisticDetector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => {
    // The official Tasks API returns one array per detected person/face/hand.
    // Holistic is configured for one signer, so index 0 is the synchronized
    // face, pose, and hand result for the current video frame.
    leftHandLandmarks: Landmark[][];
    rightHandLandmarks: Landmark[][];
    faceLandmarks: Landmark[][];
    poseLandmarks: Landmark[][];
    faceBlendshapes: Array<{ categories: Array<{ categoryName: string; score: number }> }>;
  };
  close: () => void;
};

type VisionRuntime = {
  FilesetResolver: { forVisionTasks: (root: string) => Promise<unknown> };
  HandLandmarker: {
    HAND_CONNECTIONS: unknown;
    createFromOptions: (vision: unknown, options: object) => Promise<HandDetector>;
  };
  FaceLandmarker: {
    FACE_LANDMARKS_CONTOURS: unknown;
    createFromOptions: (vision: unknown, options: object) => Promise<FaceDetector>;
  };
  PoseLandmarker: {
    POSE_CONNECTIONS: unknown;
    createFromOptions: (vision: unknown, options: object) => Promise<PoseDetector>;
  };
  HolisticLandmarker: { createFromOptions: (vision: unknown, options: object) => Promise<HolisticDetector> };
  DrawingUtils: new (context: CanvasRenderingContext2D) => {
    drawConnectors: (landmarks: Landmark[], connections: unknown, style: object) => void;
    drawLandmarks: (landmarks: Landmark[], style: object) => void;
  };
};

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

type HandTrackerProps = {
  onContextChange?: (state: SignContextState) => void;
  onContextCaptured?: (sequence: number[][], frames: string[]) => void;
  onAslStateChange?: (state: ASLSessionState) => void;
  aslEngineRef?: React.MutableRefObject<{
    backspace: () => string;
    addSpace: () => string;
    clearText: () => void;
  } | null>;
  /** Optional developer capture hook for verified, gloss-labelled samples. */
  trainingRecorder?: SignRecorder;
};

async function createHandDetector(runtime: VisionRuntime, vision: unknown): Promise<HandDetector> {
  const options = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: modelPath, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.45,
    minHandPresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });

  try {
    return await runtime.HandLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    return runtime.HandLandmarker.createFromOptions(vision, options('CPU'));
  }
}

async function createFaceDetector(runtime: VisionRuntime, vision: unknown): Promise<FaceDetector> {
  const options = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: faceModelPath, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  try {
    return await runtime.FaceLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    return runtime.FaceLandmarker.createFromOptions(vision, options('CPU'));
  }
}

async function createPoseDetector(runtime: VisionRuntime, vision: unknown): Promise<PoseDetector> {
  const options = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: poseModelPath, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
  try {
    return await runtime.PoseLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    return runtime.PoseLandmarker.createFromOptions(vision, options('CPU'));
  }
}

async function createHolisticDetector(runtime: VisionRuntime, vision: unknown): Promise<HolisticDetector> {
  const options = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: holisticModelPath, delegate },
    runningMode: 'VIDEO',
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.4,
    minFacePresenceConfidence: 0.4,
    minPoseDetectionConfidence: 0.4,
    minPosePresenceConfidence: 0.4,
    minHandLandmarksConfidence: 0.4,
  });
  try {
    return await runtime.HolisticLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    return runtime.HolisticLandmarker.createFromOptions(vision, options('CPU'));
  }
}

let visionRuntimePromise: Promise<VisionRuntime> | undefined;

function getVisionRuntime(): Promise<VisionRuntime> {
  if (!visionRuntimePromise) {
    const importFromUrl = new Function('url', 'return import(url);') as (url: string) => Promise<VisionRuntime>;
    visionRuntimePromise = importFromUrl(visionModule);
  }
  return visionRuntimePromise;
}

const FACE_INDICES = [
  61,185,40,39,37,0,267,269,270,409,291,146,91,181,84,17,314,405,321,375,
  78,191,80,81,82,13,312,311,310,415,308,95,88,178,87,14,317,402,318,324,
  1,2,98,327,4,5,195,197,33,7,163,144,145,153,154,155,133,
  362,382,381,380,374,373,390,249,263,46,53,52,65,55,276,283,282,295,285,
];
const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27];

function flattenLandmarks(points: Landmark[] | null, indices: number[]) {
  const output = Array<number>(indices.length * 3).fill(0);
  indices.forEach((sourceIndex, index) => {
    const point = points?.[sourceIndex];
    if (!point) return;
    output[index * 3] = point.x;
    output[index * 3 + 1] = point.y;
    output[index * 3 + 2] = point.z ?? 0;
  });
  return output;
}

function createMultimodalFrame(
  hands: Landmark[][],
  handednesses: Handedness | undefined,
  face: ReturnType<FaceDetector['detectForVideo']> | undefined,
  pose: ReturnType<PoseDetector['detectForVideo']> | undefined,
) {
  let left: Landmark[] | null = null;
  let right: Landmark[] | null = null;
  hands.forEach((points, index) => {
    const label = handednesses?.[index]?.[0]?.categoryName;
    if (label === 'Left') left = points;
    else if (label === 'Right') right = points;
    else if (!right) right = points;
    else if (!left) left = points;
  });

  const facePoints = face?.faceLandmarks[0] ?? null;
  const posePoints = pose?.landmarks[0] ?? null;
  const allHandIndices = Array.from({ length: 21 }, (_, index) => index);

  // Exact 390-value PopSign training layout: left hand, right hand,
  // 76 selected face landmarks, then 12 upper-body pose landmarks.
  return [
    ...flattenLandmarks(left, allHandIndices),
    ...flattenLandmarks(right, allHandIndices),
    ...flattenLandmarks(facePoints, FACE_INDICES),
    ...flattenLandmarks(posePoints, POSE_INDICES),
  ];
}

function createCtcFeatureFrame(
  hands: Landmark[][],
  handednesses: Handedness | undefined,
  face: ReturnType<FaceDetector['detectForVideo']> | undefined,
  pose: ReturnType<PoseDetector['detectForVideo']> | undefined,
) {
  let left: Landmark[] | null = null;
  let right: Landmark[] | null = null;
  hands.forEach((points, index) => {
    const label = handednesses?.[index]?.[0]?.categoryName;
    if (label === 'Left') left = points;
    else if (label === 'Right') right = points;
    else if (!right) right = points;
    else if (!left) left = points;
  });

  const allHandIndices = Array.from({ length: 21 }, (_, index) => index);
  const allPoseIndices = Array.from({ length: 33 }, (_, index) => index);
  const facePoints = face?.faceLandmarks[0] ?? null;
  const posePoints = pose?.landmarks[0] ?? null;
  const blendshapes = Array<number>(52).fill(0);
  face?.faceBlendshapes?.[0]?.categories
    .slice()
    .sort((a, b) => a.categoryName.localeCompare(b.categoryName))
    .slice(0, 52)
    .forEach((shape, index) => { blendshapes[index] = shape.score; });

  // The causal CTC Transformer consumes this stable layout:
  // hands (126) + pose (99) + face blendshapes (52) + masks (4) = 281.
  return [
    ...flattenLandmarks(left, allHandIndices),
    ...flattenLandmarks(right, allHandIndices),
    ...flattenLandmarks(posePoints, allPoseIndices),
    ...blendshapes,
    left ? 1 : 0,
    right ? 1 : 0,
    facePoints ? 1 : 0,
    posePoints ? 1 : 0,
  ];
}

function copyRawLandmarks(points: Landmark[] | null | undefined) {
  return points?.map((point) => ({ x: point.x, y: point.y, z: point.z ?? 0 })) ?? null;
}

/**
 * Keep the original Holistic coordinates alongside the compressed 281-vector.
 * MediaPipe Tasks exposes point coordinates but not RTM WholeBody-style
 * per-keypoint scores, so the export records that limitation rather than
 * manufacturing confidence values for a different pretrained model.
 */
function createRawHolisticFrame(
  hands: Landmark[][],
  handednesses: Handedness | undefined,
  face: ReturnType<FaceDetector['detectForVideo']> | undefined,
  pose: ReturnType<PoseDetector['detectForVideo']> | undefined,
  timestamp: number,
): RawHolisticFrame {
  let left: Landmark[] | null = null;
  let right: Landmark[] | null = null;
  hands.forEach((points, index) => {
    const label = handednesses?.[index]?.[0]?.categoryName;
    if (label === 'Left') left = points;
    else if (label === 'Right') right = points;
  });
  return {
    timestamp_ms: timestamp,
    left_hand: copyRawLandmarks(left),
    right_hand: copyRawLandmarks(right),
    face: copyRawLandmarks(face?.faceLandmarks[0]),
    pose: copyRawLandmarks(pose?.landmarks[0]),
    confidence_representation: 'not_available_from_mediapipe_tasks',
  };
}

function distanceBetweenWrist(a: Landmark[], b: Landmark[]) {
  const dx = a[0]?.x - b[0]?.x;
  const dy = a[0]?.y - b[0]?.y;
  return (dx * dx) + (dy * dy);
}

function stabilizeHandOrder(current: Landmark[][], previous: Landmark[][]) {
  if (current.length !== 2 || previous.length !== 2) return current;

  const inOrder = distanceBetweenWrist(current[0], previous[0]) + distanceBetweenWrist(current[1], previous[1]);
  const swapped = distanceBetweenWrist(current[0], previous[1]) + distanceBetweenWrist(current[1], previous[0]);
  return swapped < inOrder ? [current[1], current[0]] : current;
}

function smoothLandmarks(current: Landmark[][], previous: Landmark[][], alpha = 0.8) {
  const ordered = stabilizeHandOrder(current, previous);
  return ordered.map((hand, handIndex) => hand.map((point, landmarkIndex) => {
    const prior = previous[handIndex]?.[landmarkIndex];
    if (!prior) return point;
    return {
      x: prior.x + (point.x - prior.x) * alpha,
      y: prior.y + (point.y - prior.y) * alpha,
      z: (prior.z ?? 0) + ((point.z ?? 0) - (prior.z ?? 0)) * alpha,
    };
  }));
}

export default function HandTracker({
  onContextChange,
  onContextCaptured,
  onAslStateChange,
  aslEngineRef,
  trainingRecorder,
}: HandTrackerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sequenceRef = useRef<number[][]>([]);
  const frameImagesRef = useRef<string[]>([]);
  const smoothedHandsRef = useRef<Landmark[][]>([]);
  const totalTrackedFramesRef = useRef(0);
  const lastLiveUpdateAtRef = useRef(-Infinity);
  const onContextChangeRef = useRef(onContextChange);
  const onContextCapturedRef = useRef(onContextCaptured);
  const onAslStateChangeRef = useRef(onAslStateChange);
  const latestWordStateRef = useRef<ASLWordState | undefined>(undefined);

  // Initialize ASL Gesture Engine for zero-latency on-device processing
  const aslEngine = useRef<ASLGestureEngine>(
    new ASLGestureEngine({
      minConfidence: 0.62,
      holdFramesRequired: 7,
      onStateChange: (state) => {
        onAslStateChangeRef.current?.(state);
      },
    }),
  ).current;
  const wordEngine = useRef<ASLWordEngine>(
    new ASLWordEngine((state) => {
      latestWordStateRef.current = state;
    }),
  ).current;

  // Expose controller methods to parent via ref
  useEffect(() => {
    if (aslEngineRef) {
      aslEngineRef.current = {
        backspace: () => aslEngine.backspace(),
        addSpace: () => aslEngine.addSpace(),
        clearText: () => {
          aslEngine.clearText();
          wordEngine.clear();
        },
      };
    }
  }, [aslEngineRef, aslEngine, wordEngine]);

  useEffect(() => {
    onContextChangeRef.current = onContextChange;
  }, [onContextChange]);

  useEffect(() => {
    onContextCapturedRef.current = onContextCaptured;
  }, [onContextCaptured]);

  useEffect(() => {
    onAslStateChangeRef.current = onAslStateChange;
  }, [onAslStateChange]);

  useEffect(() => {
    let active = true;
    let stream: MediaStream | undefined;
    let animationFrame: number | undefined;
    let detector: HandDetector | undefined;
    let faceDetector: FaceDetector | undefined;
    let poseDetector: PoseDetector | undefined;
    let holisticDetector: HolisticDetector | undefined;
    let lastContextState = '';

    const reportContext = (
      handCount: number,
      error?: string,
      aslState?: ASLSessionState,
      faceTracked = false,
      poseTracked = false,
    ) => {
      const frameCount = sequenceRef.current.length;
      const next: SignContextState = {
        frameCount,
        maxFrames: maxContextFrames,
        handCount,
        ready: frameCount >= minimumFramesForTranslation,
        error,
        aslState,
        faceTracked,
        poseTracked,
        wordState: latestWordStateRef.current,
      };
      const wordState = latestWordStateRef.current;
      const key = `${frameCount}:${handCount}:${error ?? ''}:${faceTracked}:${poseTracked}:${aslState?.currentLetter ?? ''}:${aslState?.spelledText ?? ''}:${Math.round((aslState?.holdProgress ?? 0) * 10)}:${wordState?.currentWord ?? ''}:${wordState?.phrase ?? ''}:${Math.round((wordState?.currentConfidence ?? 0) * 20)}`;
      if (key !== lastContextState) {
        lastContextState = key;
        onContextChangeRef.current?.(next);
      }
    };

    async function initialise() {
      try {
        const runtime = await getVisionRuntime();
        const vision = await runtime.FilesetResolver.forVisionTasks(wasmRoot);
        let fallbackSetup: Promise<void> | undefined;
        const activateIndividualFallback = () => {
          if (!fallbackSetup) {
            fallbackSetup = (async () => {
              const [hand, face, pose] = await Promise.allSettled([
                createHandDetector(runtime, vision),
                createFaceDetector(runtime, vision),
                createPoseDetector(runtime, vision),
              ]);
              if (hand.status !== 'fulfilled') {
                throw hand.reason instanceof Error ? hand.reason : new Error('Hand tracker could not start.');
              }
              detector = hand.value;
              faceDetector = face.status === 'fulfilled' ? face.value : undefined;
              poseDetector = pose.status === 'fulfilled' ? pose.value : undefined;
            })();
          }
          return fallbackSetup;
        };
        // Holistic produces hands, face, blendshapes, and pose from one
        // timestamped frame. This is essential for temporal signing motion.
        // Some devices/browsers cannot run the composite model, so retain the
        // existing individual-task implementation as a complete fallback.
        try {
          holisticDetector = await createHolisticDetector(runtime, vision);
        } catch (holisticError) {
          console.warn('Holistic tracker unavailable; using individual MediaPipe trackers.', holisticError);
          await activateIndividualFallback();
        }

        if (!active) return;

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) throw new Error('Camera preview could not be prepared.');

        video.srcObject = stream;
        await video.play();

        const context = canvas.getContext('2d');
        if (!context) throw new Error('Landmark canvas is not available.');
        const drawingUtils = new runtime.DrawingUtils(context);
        let lastVideoTime = -1;
        reportContext(0);

        const track = () => {
          if (!active) return;
          // A failed Holistic frame can trigger an asynchronous fallback setup.
          // Keep the animation loop alive until the individual trackers are ready.
          if (!detector && !holisticDetector) {
            animationFrame = requestAnimationFrame(track);
            return;
          }

          if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            if (video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime;
              const timestamp = performance.now();
              let results: { landmarks: Landmark[][]; handednesses?: Handedness; handedness?: Handedness };
              let face: ReturnType<FaceDetector['detectForVideo']> | undefined;
              let pose: ReturnType<PoseDetector['detectForVideo']> | undefined;

              try {
                if (detector) {
                  results = detector.detectForVideo(video, timestamp);
                  // A face or pose model failure must not stop hand/motion
                  // capture. Preserve the available modalities for this frame.
                  try { face = faceDetector?.detectForVideo(video, timestamp); }
                  catch (error) { console.warn('Face tracker frame failed.', error); }
                  try { pose = poseDetector?.detectForVideo(video, timestamp); }
                  catch (error) { console.warn('Pose tracker frame failed.', error); }
                } else if (holisticDetector) {
                  const holistic = holisticDetector.detectForVideo(video, timestamp);
                  const landmarks: Landmark[][] = [];
                  const handednesses: Handedness = [];
                  const leftHand = holistic.leftHandLandmarks[0];
                  const rightHand = holistic.rightHandLandmarks[0];
                  if (leftHand && leftHand.length > 0) {
                    landmarks.push(leftHand);
                    handednesses.push([{ categoryName: 'Left', score: 1 }]);
                  }
                  if (rightHand && rightHand.length > 0) {
                    landmarks.push(rightHand);
                    handednesses.push([{ categoryName: 'Right', score: 1 }]);
                  }
                  results = { landmarks, handednesses, handedness: handednesses };
                  face = {
                    faceLandmarks: holistic.faceLandmarks,
                    faceBlendshapes: holistic.faceBlendshapes,
                  };
                  pose = { landmarks: holistic.poseLandmarks };
                } else {
                  animationFrame = requestAnimationFrame(track);
                  return;
                }
              } catch (error) {
                if (holisticDetector) {
                  console.warn('Holistic tracking frame failed; switching to individual trackers.', error);
                  holisticDetector.close();
                  holisticDetector = undefined;
                  void activateIndividualFallback().catch((fallbackError) => {
                    reportContext(0, `Tracking fallback failed: ${fallbackError instanceof Error ? fallbackError.message : 'unknown error'}`);
                  });
                  reportContext(0, 'Reconnecting hand, face, and body tracking…');
                } else {
                  reportContext(0, `Hand tracking frame failed: ${error instanceof Error ? error.message : 'unknown error'}`);
                }
                animationFrame = requestAnimationFrame(track);
                return;
              }
              context.clearRect(0, 0, canvas.width, canvas.height);

              const smoothed = smoothLandmarks(results.landmarks, smoothedHandsRef.current);
              smoothedHandsRef.current = smoothed;

              let currentAslState: ASLSessionState | undefined;

              if (smoothed.length === 0) {
                smoothedHandsRef.current = [];
                currentAslState = aslEngine.processFrame(undefined);
                wordEngine.processFrame(undefined);
              } else {
                // Perform instant on-device ASL classification (<0.2ms)
                const primaryHand = smoothed[0];
                const handednessCategory =
                  results.handednesses?.[0]?.[0]?.categoryName ??
                  results.handedness?.[0]?.[0]?.categoryName ??
                  'Right';

                currentAslState = aslEngine.processFrame(primaryHand, handednessCategory);

                const handednessesArray = results.handedness ?? results.handednesses;
                const multimodalFrame = createMultimodalFrame(smoothed, handednessesArray, face, pose);
                wordEngine.processFrame(multimodalFrame, smoothed, face?.faceLandmarks?.[0], pose?.landmarks?.[0]);
                const sequence = sequenceRef.current;
                const temporalFrame = createCtcFeatureFrame(smoothed, handednessesArray, face, pose);
                sequence.push(temporalFrame);
                if (trainingRecorder && temporalFrame.length === TEMPORAL_FEATURE_COUNT) {
                  trainingRecorder.addFrame(temporalFrame, timestamp);
                  trainingRecorder.addRawHolisticFrame(createRawHolisticFrame(
                    smoothed,
                    handednessesArray,
                    face,
                    pose,
                    timestamp,
                  ));
                }
                if (sequence.length > maxContextFrames) sequence.shift();
                totalTrackedFramesRef.current += 1;
                if (totalTrackedFramesRef.current % snapshotIntervalFrames === 0) {
                  const snapshot = document.createElement('canvas');
                  const width = 512;
                  snapshot.width = width;
                  snapshot.height = Math.round(width * video.videoHeight / video.videoWidth);
                  snapshot.getContext('2d')?.drawImage(video, 0, 0, snapshot.width, snapshot.height);
                  frameImagesRef.current.push(snapshot.toDataURL('image/jpeg', 0.68));
                  if (frameImagesRef.current.length > snapshotCount) frameImagesRef.current.shift();
                }
                const now = performance.now();
                if (
                  sequence.length >= minimumFramesForTranslation
                  && frameImagesRef.current.length === snapshotCount
                  && now - lastLiveUpdateAtRef.current >= liveUpdateIntervalMs
                ) {
                  lastLiveUpdateAtRef.current = now;
                  onContextCapturedRef.current?.(sequence.map((frame) => [...frame]), [...frameImagesRef.current]);
                }
              }

              const faceLandmarks = face?.faceLandmarks[0];
              const poseLandmarks = pose?.landmarks[0];
              reportContext(smoothed.length, undefined, currentAslState, Boolean(faceLandmarks), Boolean(poseLandmarks));

              // Face and upper-body cues make temporal signing models more
              // expressive. Keep their rendering lightweight so hand landmarks
              // remain the primary visual focus.
              if (faceLandmarks) {
                drawingUtils.drawConnectors(faceLandmarks, runtime.FaceLandmarker.FACE_LANDMARKS_CONTOURS, {
                  color: 'rgba(108, 212, 255, 0.88)',
                  lineWidth: 1.4,
                });
                drawingUtils.drawLandmarks(faceLandmarks, {
                  color: 'rgba(174, 235, 255, 0.75)',
                  radius: 1.15,
                });
              }
              if (poseLandmarks) {
                drawingUtils.drawConnectors(poseLandmarks, runtime.PoseLandmarker.POSE_CONNECTIONS, {
                  color: 'rgba(255, 190, 92, 0.95)',
                  lineWidth: 3.25,
                });
                drawingUtils.drawLandmarks(poseLandmarks, {
                  color: '#FFD28A',
                  radius: 2.4,
                });
              }

              // Render landmark connections & joints
              for (const landmarks of smoothed) {
                drawingUtils.drawConnectors(landmarks, runtime.HandLandmarker.HAND_CONNECTIONS, {
                  color: 'rgba(2, 18, 10, 0.94)',
                  lineWidth: 12,
                });
                drawingUtils.drawConnectors(landmarks, runtime.HandLandmarker.HAND_CONNECTIONS, {
                  color: '#D6FF74',
                  lineWidth: 5,
                });
                drawingUtils.drawLandmarks(landmarks, {
                  color: 'rgba(2, 18, 10, 0.94)',
                  radius: 5.5,
                });
                drawingUtils.drawLandmarks(landmarks, {
                  color: '#FF826F',
                  radius: 3.25,
                });
              }

              // Render interactive targeting ring around wrist
              if (currentAslState && currentAslState.currentLetter && smoothed.length > 0) {
                const wrist = smoothed[0][0];
                if (wrist) {
                  const wx = wrist.x * canvas.width;
                  const wy = wrist.y * canvas.height;
                  context.beginPath();
                  context.arc(wx, wy, 24 + currentAslState.holdProgress * 12, 0, 2 * Math.PI);
                  context.strokeStyle = currentAslState.isHolding ? '#C8FF72' : 'rgba(200, 255, 114, 0.45)';
                  context.lineWidth = 3.5;
                  context.stroke();
                }
              }
            }
          }

          animationFrame = requestAnimationFrame(track);
        };

        track();
      } catch (error) {
        if (active) {
          const message = error instanceof Error ? error.message : 'Unable to start hand tracking.';
          reportContext(0, `Hand tracking unavailable: ${message}`);
        }
      }
    }

    initialise();

    return () => {
      active = false;
      if (animationFrame !== undefined) cancelAnimationFrame(animationFrame);
      stream?.getTracks().forEach((track) => track.stop());
      detector?.close();
      faceDetector?.close();
      poseDetector?.close();
      holisticDetector?.close();
    };
  }, [aslEngine, wordEngine]);

  return (
    <div aria-label="Live sign language hand tracking" style={styles.container}>
      <video ref={videoRef} autoPlay playsInline muted style={styles.video} />
      <canvas ref={canvasRef} style={styles.canvas} />
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    position: 'absolute',
    inset: 0,
    overflow: 'hidden',
    background: '#07110F',
  },
  video: {
    width: '100%',
    height: '100%',
    display: 'block',
    objectFit: 'contain',
    transform: 'scaleX(-1)',
    transformOrigin: 'center',
  },
  canvas: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    transform: 'scaleX(-1)',
    transformOrigin: 'center',
    filter: 'drop-shadow(0 0 5px rgba(200,255,114,.68))',
    pointerEvents: 'none',
  },
};

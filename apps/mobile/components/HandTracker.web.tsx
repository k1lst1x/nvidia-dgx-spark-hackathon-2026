import { CSSProperties, useEffect, useRef } from 'react';

const wasmRoot = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/wasm';
const modelPath = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const visionModule = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/vision_bundle.mjs';
const framesPerGesture = 30;
const maxContextFrames = framesPerGesture * 3;

type Landmark = { x: number; y: number; z?: number };
type HandDetector = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => { landmarks: Landmark[][] };
  close: () => void;
};

type VisionRuntime = {
  FilesetResolver: { forVisionTasks: (root: string) => Promise<unknown> };
  HandLandmarker: {
    HAND_CONNECTIONS: unknown;
    createFromOptions: (vision: unknown, options: object) => Promise<HandDetector>;
  };
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
};

type HandTrackerProps = {
  onContextChange?: (state: SignContextState) => void;
  onContextCaptured?: (sequence: number[][], frames: string[]) => void;
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

let visionRuntimePromise: Promise<VisionRuntime> | undefined;

function getVisionRuntime(): Promise<VisionRuntime> {
  if (!visionRuntimePromise) {
    const importFromUrl = new Function('url', 'return import(url);') as (url: string) => Promise<VisionRuntime>;
    visionRuntimePromise = importFromUrl(visionModule);
  }
  return visionRuntimePromise;
}

function createLandmarkFrame(hands: Landmark[][]) {
  const frame = Array<number>(126).fill(0);

  hands.forEach((hand, handIndex) => {
    if (handIndex >= 2) return;

    hand.forEach((point, landmarkIndex) => {
      const offset = handIndex * 63 + landmarkIndex * 3;
      frame[offset] = point.x;
      frame[offset + 1] = point.y;
      frame[offset + 2] = point.z ?? 0;
    });
  });

  return frame;
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

export default function HandTracker({ onContextChange, onContextCaptured }: HandTrackerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sequenceRef = useRef<number[][]>([]);
  const frameImagesRef = useRef<string[]>([]);
  const maxHandCountRef = useRef(0);
  const smoothedHandsRef = useRef<Landmark[][]>([]);
  const contextCapturedRef = useRef(false);
  const onContextChangeRef = useRef(onContextChange);
  const onContextCapturedRef = useRef(onContextCaptured);

  useEffect(() => {
    onContextChangeRef.current = onContextChange;
  }, [onContextChange]);

  useEffect(() => {
    onContextCapturedRef.current = onContextCaptured;
  }, [onContextCaptured]);

  useEffect(() => {
    let active = true;
    let stream: MediaStream | undefined;
    let animationFrame: number | undefined;
    let detector: HandDetector | undefined;
    let lastContextState = '';

    const reportContext = (handCount: number, error?: string) => {
      maxHandCountRef.current = Math.max(maxHandCountRef.current, handCount);
      const frameCount = sequenceRef.current.length;
      const next: SignContextState = {
        frameCount,
        maxFrames: maxContextFrames,
        handCount: frameCount === maxContextFrames ? maxHandCountRef.current : handCount,
        ready: frameCount === maxContextFrames,
        error,
      };
      const key = `${frameCount}:${handCount}:${error ?? ''}`;
      if (key !== lastContextState) {
        lastContextState = key;
        onContextChangeRef.current?.(next);
      }
    };

    async function initialise() {
      try {
        const runtime = await getVisionRuntime();
        const vision = await runtime.FilesetResolver.forVisionTasks(wasmRoot);
        detector = await createHandDetector(runtime, vision);

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
          if (!active || !detector) return;

          if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            if (video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime;
              const results = detector.detectForVideo(video, performance.now());
              context.clearRect(0, 0, canvas.width, canvas.height);

              const smoothed = smoothLandmarks(results.landmarks, smoothedHandsRef.current);
              smoothedHandsRef.current = smoothed;

              if (smoothed.length === 0) {
                smoothedHandsRef.current = [];
              } else if (!contextCapturedRef.current) {
                const sequence = sequenceRef.current;
                sequence.push(createLandmarkFrame(smoothed));
                if (sequence.length > maxContextFrames) sequence.shift();
                if ([30, 60, 90].includes(sequence.length) && frameImagesRef.current.length < 3) {
                  const snapshot = document.createElement('canvas');
                  const width = 512;
                  snapshot.width = width;
                  snapshot.height = Math.round(width * video.videoHeight / video.videoWidth);
                  snapshot.getContext('2d')?.drawImage(video, 0, 0, snapshot.width, snapshot.height);
                  frameImagesRef.current.push(snapshot.toDataURL('image/jpeg', 0.68));
                }
                if (sequence.length === maxContextFrames) {
                  contextCapturedRef.current = true;
                  onContextCapturedRef.current?.(sequence.map((frame) => [...frame]), [...frameImagesRef.current]);
                }
              }
              reportContext(smoothed.length);

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
    };
  }, []);

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
    objectFit: 'cover',
    transform: 'scaleX(-1)',
  },
  canvas: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transform: 'scaleX(-1)',
    filter: 'drop-shadow(0 0 5px rgba(200,255,114,.68))',
    pointerEvents: 'none',
  },
};

import type { CameraView } from 'expo-camera';
import { RefObject, useCallback, useEffect, useRef, useState } from 'react';
import { detectObjects, detectVideoFrame, localDetectorBackend, warmUpDetector } from './detector';
import type { DetectionResult, LocalDetection } from './detector.types';
import type { BlockedView } from './blockedView';

/** Directions the local camera frame looks blocked in, and when it was seen. */
export type BlockedFrame = BlockedView & { at: number };

export type AssistMode = 'scene' | 'sign';

export type VisionObject = {
  label: string;
  confidence: number;
  position: string;
  x?: number | null;
  y?: number | null;
  width?: number | null;
  height?: number | null;
  distance_m?: number | null;
  relative_proximity?: number | null;
  track_id?: number | null;
  motion?: {
    trend: 'approaching' | 'receding' | 'stable' | 'unknown';
    direction: 'left' | 'ahead' | 'right';
    distance_m?: number | null;
    relative_depth_velocity_mps?: number | null;
    time_to_closest_approach_s?: number | null;
    minimum_separation_m?: number | null;
    confidence: number;
    sample_count: number;
    camera_relative: boolean;
    world_motion_status: string;
    world_speed_mps?: number | null;
    experimental: true;
  } | null;
  spatial?: {
    estimate?: boolean;
    depth_source?: string;
    mask_coverage?: number;
    center_m?: number[];
    extent_m?: number[];
    point_count?: number;
    calibrated_camera?: boolean;
  } | null;
};

export type ApproachingWarning = { label: string; distance_m: number };

export type CameraCoachResult = {
  blur: boolean;
  poor_lighting: boolean;
  overexposed: boolean;
  cropped_labels: string[];
  sharpness_score: number;
  brightness_score: number;
};

export type VisionResult = {
  summary: string;
  speech: string;
  scene_key: string;
  urgency: 'low' | 'medium' | 'high';
  changed: boolean;
  objects: VisionObject[];
  task: AssistMode;
  should_speak?: boolean;
  approaching?: ApproachingWarning[];
};

type TrackingResult = {
  objects: VisionObject[];
  tracking_latency_ms: number;
  detector: 'yolo11-botsort' | 'disabled';
  metric_depth: boolean;
  camera_coach: CameraCoachResult | null;
  experimental: true;
  collision_warnings_enabled: false;
  obstacle_points?: number[][];
  obstacle_range_m?: number | null;
};

/** Camera-frame [x right, y down, z forward] meter dots for unlabeled surfaces. */
export type ObstacleFrame = {
  /** Increments once per tracking response so the map ingests each frame once. */
  sequence: number;
  capturedAt: number;
  points: Array<[number, number, number]>;
  rangeM: number | null;
};

export type SignTranslation = {
  transcript: string;
  segment: string;
  confidence: number;
  is_final: boolean;
  should_speak: boolean;
};

type VisionState = 'starting' | 'analyzing' | 'live' | 'offline' | 'paused';

export type VisionPerformance = {
  capture_fps: number;
  capture_ms: number | null;
  local_detection_ms: number | null;
  tracking_latency_ms: number | null;
  tracking_age_ms: number | null;
  tracking_status: 'waiting' | 'live' | 'slow' | 'offline';
  /** Where the local detector runs: the GPU (WebGPU), the CPU (wasm), or native. */
  local_backend?: 'webgpu' | 'wasm' | null;
};

// Local object frames should follow camera movement; cloud narration can update less often.
const LOCAL_SCAN_INTERVAL_MS = 200;
// While the browser detects straight from the video, the scan loop only
// decides when the server and cloud need a new photo.
const LIVE_SCAN_INTERVAL_MS = 250;
const LIVE_LOCAL_FRESH_MS = 1000;

function detectionSignature(detections: LocalDetection[]) {
  return detections
    .filter((item) => item.confidence >= 0.5)
    .map((item) => item.label)
    .sort()
    .join('|');
}

/** The camera preview's <video> element on web, once it is playing. */
function findCameraVideo(): HTMLVideoElement | null {
  if (typeof document === 'undefined') return null;
  for (const video of Array.from(document.querySelectorAll('video'))) {
    if (video.readyState >= 2 && video.videoWidth > 0) return video;
  }
  return null;
}

const nextAnimationFrame = () => new Promise<void>((resolve) => {
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
  else setTimeout(resolve, 16);
});
const LOCAL_API_PORT = '8000';

function isLoopbackHost(hostname: string): boolean {
  return ['localhost', '127.0.0.1', '::1', '[::1]'].includes(hostname);
}

export function isSecureEndpoint(urlStr: string): boolean {
  try {
    const parsed = new URL(urlStr);
    if (parsed.protocol === 'https:') return true;
    if (parsed.protocol === 'http:' && isLoopbackHost(parsed.hostname)) return true;
    return false;
  } catch {
    return false;
  }
}

function normalizeApiUrl(value: string): string {
  const trimmed = value.trim().replace(/\/$/, '');
  try {
    const parsed = new URL(trimmed);
    // 8082 is Expo Web in this project; FastAPI listens on 8000.
    if (isLoopbackHost(parsed.hostname) && parsed.port === '8082') {
      parsed.port = LOCAL_API_PORT;
      return parsed.toString().replace(/\/$/, '');
    }
    // Enforce transport security: remote endpoints must use https
    if (parsed.protocol === 'http:' && !isLoopbackHost(parsed.hostname)) {
      return '';
    }
  } catch {
    // Preserve validation behavior for malformed custom values.
  }
  return trimmed;
}

export function getApiUrl(): string {
  if (typeof window !== 'undefined' && window.localStorage) {
    const custom = window.localStorage.getItem('guidelens_api_url');
    if (custom && custom.trim()) {
      const normalized = normalizeApiUrl(custom);
      if (normalized !== custom.trim().replace(/\/$/, '')) window.localStorage.setItem('guidelens_api_url', normalized);
      return normalized;
    }
  }
  const configured = process.env.EXPO_PUBLIC_API_URL ? normalizeApiUrl(process.env.EXPO_PUBLIC_API_URL) : '';
  if (configured) return configured;
  if (typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname)) {
    return 'http://localhost:8000';
  }
  return '';
}

const configuredApiUrl = process.env.EXPO_PUBLIC_API_URL ? normalizeApiUrl(process.env.EXPO_PUBLIC_API_URL) : '';
const API_URL = configuredApiUrl || (
  typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname)
    ? 'http://localhost:8000'
    : ''
);
export const hasVisionBackend = Boolean(API_URL);
const sessionId = `gl-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

function requireApiUrl() {
  const url = getApiUrl();
  if (!url) throw new Error('EXPO_PUBLIC_API_URL is not configured. Configure it in Settings or set EXPO_PUBLIC_API_URL.');
  return url;
}

function dataUriToBlob(uri: string): Blob {
  const parts = uri.split(',');
  const mime = parts[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
  const binary = atob(parts[1]);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    array[i] = binary.charCodeAt(i);
  }
  return new Blob([array], { type: mime });
}

async function analyzeFrame(uri: string, mode: AssistMode, previousSceneKey: string, signal: AbortSignal) {
  const apiUrl = requireApiUrl();
  const data = new FormData();
  // Browsers require a real Blob in multipart form data; the React Native
  // `{ uri, name, type }` object otherwise becomes a text field and FastAPI
  // returns 422 before the vision model ever sees the image.
  if (typeof window !== 'undefined') {
    let image: Blob;
    if (uri.startsWith('data:')) {
      image = dataUriToBlob(uri);
    } else {
      const imageResponse = await fetch(uri);
      if (!imageResponse.ok) throw new Error('Camera frame could not be prepared for analysis');
      image = await imageResponse.blob();
    }
    data.append('image', image, 'frame.jpg');
  } else {
    data.append('image', { uri, name: 'frame.jpg', type: 'image/jpeg' } as unknown as Blob);
  }
  data.append('task', mode);
  data.append('previous_scene_key', previousSceneKey);
  let response: Response;
  try {
    response = await fetch(`${apiUrl}/v1/describe`, {
      method: 'POST', body: data, headers: { 'X-Session-ID': sessionId }, signal,
    });
  } catch {
    throw new Error('Vision server unavailable');
  }
  if (!response.ok) {
    const retryAfter = Number(response.headers.get('retry-after') || 0);
    const error = new Error(`Vision request failed: ${response.status}`) as Error & { retryAfter?: number };
    error.retryAfter = retryAfter;
    throw error;
  }
  return (await response.json()) as VisionResult;
}

async function trackFrame(uri: string, signal: AbortSignal): Promise<TrackingResult> {
  const apiUrl = requireApiUrl();
  const data = new FormData();
  if (typeof window !== 'undefined') {
    let image: Blob;
    if (uri.startsWith('data:')) {
      image = dataUriToBlob(uri);
    } else {
      const imageResponse = await fetch(uri);
      if (!imageResponse.ok) throw new Error('Camera frame could not be prepared for tracking');
      image = await imageResponse.blob();
    }
    data.append('image', image, 'frame.jpg');
  } else {
    data.append('image', { uri, name: 'frame.jpg', type: 'image/jpeg' } as unknown as Blob);
  }
  const response = await fetch(`${apiUrl}/v1/track`, {
    method: 'POST', body: data, headers: { 'X-Session-ID': sessionId }, signal,
  });
  if (!response.ok) throw new Error(`Tracking request failed: ${response.status}`);
  return await response.json() as TrackingResult;
}

export async function interpretSignSequence(frames: string[], previousTranscript = '', signal?: AbortSignal) {
  const apiUrl = requireApiUrl();
  if (frames.length !== 3) throw new Error('Three sign-language context frames are required');
  let response: Response;
  try {
    response = await fetch(`${apiUrl}/v1/interpret-sign`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Session-ID': sessionId },
      body: JSON.stringify({ frames, previous_transcript: previousTranscript.slice(-600) }), signal,
    });
  } catch {
    throw new Error('Sign interpretation server offline');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail || `Sign interpretation failed: ${response.status}`);
  }
  return (await response.json()) as SignTranslation;
}

async function imageUriToDataUrl(uri: string) {
  if (uri.startsWith('data:')) {
    return uri.replace(/^data:[^;]*;base64,/, 'data:image/jpeg;base64,');
  }
  const response = await fetch(uri);
  if (!response.ok) throw new Error('Camera frame could not be prepared for sign translation');
  const blob = await response.blob();
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Camera frame could not be encoded'));
    reader.onload = () => {
      const encoded = String(reader.result);
      resolve(encoded.replace(/^data:[^;]*;base64,/, 'data:image/jpeg;base64,'));
    };
    reader.readAsDataURL(blob);
  });
}

// Server tracking (metric depth) answers about once a second; the local
// detector runs about five times a second. Keep the last server answer this
// long and lend its measurements to the matching local boxes in between.
const TRACKED_DEPTH_TTL_MS = 2500;

export type TrackedSnapshot = { objects: VisionObject[]; at: number };

function boxMatchScore(left: VisionObject, right: VisionObject) {
  const values = [left.x, left.y, left.width, left.height, right.x, right.y, right.width, right.height];
  if (!values.every((value) => typeof value === 'number' && Number.isFinite(value))) return 0;
  const [lx, ly, lw, lh, rx, ry, rw, rh] = values as number[];
  const overlapX = Math.max(0, Math.min(lx + lw, rx + rw) - Math.max(lx, rx));
  const overlapY = Math.max(0, Math.min(ly + lh, ry + rh) - Math.max(ly, ry));
  const intersection = overlapX * overlapY;
  const union = lw * lh + rw * rh - intersection;
  const iou = union > 0 ? intersection / union : 0;
  // The camera may have moved a little since the server's frame.
  const centerGap = Math.hypot(lx + lw / 2 - (rx + rw / 2), ly + lh / 2 - (ry + rh / 2));
  return Math.max(iou, 1 - centerGap / 0.15);
}

/**
 * Give local detections the server's measurements (distance, 3D center,
 * motion, track id) when a fresh server object with the same label overlaps.
 *
 * Without this, every local scan replaced the server's objects with copies
 * that had no distance, so the map showed "no distance" for everything. Local
 * tracker ids are dropped because they share a number space with the server's
 * BoT-SORT ids and would merge two different objects on the map.
 */
export function withTrackedDepth(objects: VisionObject[], tracked: TrackedSnapshot | null, now = Date.now()): VisionObject[] {
  const fresh = tracked && now - tracked.at <= TRACKED_DEPTH_TTL_MS ? tracked.objects : [];
  const used = new Set<number>();
  return objects.map((item) => {
    let best = -1;
    let bestScore = 0.2;
    fresh.forEach((candidate, index) => {
      if (used.has(index) || candidate.label.toLowerCase() !== item.label.toLowerCase()) return;
      const score = boxMatchScore(item, candidate);
      if (score > bestScore) {
        bestScore = score;
        best = index;
      }
    });
    if (best < 0) return { ...item, track_id: null };
    used.add(best);
    const match = fresh[best];
    return {
      ...item,
      distance_m: match.distance_m ?? null,
      relative_proximity: match.relative_proximity ?? item.relative_proximity ?? null,
      spatial: match.spatial ?? item.spatial,
      motion: match.motion ?? item.motion,
      track_id: match.track_id ?? null,
    };
  });
}

/** Where a blocked view sits, in words, or null when too little is blocked to say. */
function blockedDirection(blocked: BlockedView | null | undefined): 'ahead' | 'on the left' | 'on the right' | null {
  if (!blocked) return null;
  const centers = blocked.columns.map((value, index) => (value ? blocked.centers[index] : null)).filter((value): value is number => value != null);
  if (centers.length < 3) return null;
  const middle = centers.reduce((sum, value) => sum + value, 0) / centers.length;
  return middle < 0.38 ? 'on the left' : middle > 0.62 ? 'on the right' : 'ahead';
}

export function synthesizeLocalSceneResult(
  detections: LocalDetection[],
  previousSceneKey: string,
  mode: AssistMode,
  blocked?: BlockedView | null,
): VisionResult {
  const visionObjects: VisionObject[] = detections.map((d) => ({
    label: d.label,
    confidence: d.confidence,
    position: d.direction || 'ahead',
    x: d.x,
    y: d.y,
    width: d.width,
    height: d.height,
    // Local YOLOX has no calibrated depth source. Its class-height heuristic
    // remains internal to layout prioritization, but must not be presented as
    // a real-world range or approaching-object measurement.
    distance_m: null,
    relative_proximity: null,
    track_id: d.id ?? null,
  }));

  // Local detections are visual hints, not calibrated obstacle or collision
  // measurements. Mobility warnings stay disabled until validated separately.
  // 1. No recognized detections
  if (detections.length === 0) {
    // Nothing named is not the same as nothing there: a plain wall, a trunk or
    // a covered lens has no label. Say what the camera can tell, and never
    // call the path clear (it cannot see stairs, curbs or glass either).
    if (blocked && (blocked.covered || blocked.dark)) {
      const scene_key = blocked.dark ? 'view_dark' : 'view_covered';
      return {
        summary: "Camera can't see",
        speech: "The camera can't see. It may be covered or too dark.",
        scene_key,
        urgency: 'medium',
        changed: scene_key !== previousSceneKey,
        objects: [],
        task: mode,
        should_speak: scene_key !== previousSceneKey,
      };
    }
    const direction = blockedDirection(blocked);
    if (direction) {
      const scene_key = `view_blocked_${direction.replace(/ /g, '_')}`;
      return {
        summary: `View blocked ${direction}`,
        speech: `Something is blocking the view ${direction}. I can't tell how far.`,
        scene_key,
        urgency: 'medium',
        changed: scene_key !== previousSceneKey,
        objects: [],
        task: mode,
        should_speak: scene_key !== previousSceneKey,
      };
    }
    const scene_key = 'empty_path';
    return {
      summary: 'No objects recognized',
      speech: 'No objects recognized ahead.',
      scene_key,
      urgency: 'low',
      changed: scene_key !== previousSceneKey,
      objects: [],
      task: mode,
      should_speak: scene_key !== previousSceneKey,
    };
  }

  // 2. Natural spatial narration of detected objects
  const ahead = detections.filter((d) => d.direction === 'ahead');
  const left = detections.filter((d) => d.direction === 'left');
  const right = detections.filter((d) => d.direction === 'right');

  const parts: string[] = [];
  if (ahead.length > 0) {
    parts.push(`Ahead: ${ahead.slice(0, 2).map((d) => d.label).join(', ')}`);
  }
  if (right.length > 0) {
    parts.push(`To your right: ${right.slice(0, 2).map((d) => d.label).join(', ')}`);
  }
  if (left.length > 0) {
    parts.push(`To your left: ${left.slice(0, 2).map((d) => d.label).join(', ')}`);
  }

  const speech = parts.join('. ') + '.';
  const scene_key = detections
    .slice(0, 4)
    .map((d) => `${d.label}_${d.direction}`)
    .sort()
    .join('|');

  return {
    summary: `Nearby: ${detections.slice(0, 3).map((d) => d.label).join(', ')}`,
    speech,
    scene_key,
    urgency: 'low',
    changed: scene_key !== previousSceneKey,
    objects: visionObjects,
    task: mode,
    should_speak: scene_key !== previousSceneKey,
  };
}

export function useVisionLoop(camera: RefObject<CameraView | null>, mode: AssistMode, enabled: boolean) {
  const [result, setResult] = useState<VisionResult | null>(null);
  const [state, setState] = useState<VisionState>('starting');
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [detections, setDetections] = useState<LocalDetection[]>([]);
  const [cameraCoach, setCameraCoach] = useState<CameraCoachResult | null>(null);
  const [obstacles, setObstacles] = useState<ObstacleFrame | null>(null);
  const [blocked, setBlocked] = useState<BlockedFrame | null>(null);
  const obstacleSequence = useRef(0);
  const [performance, setPerformance] = useState<VisionPerformance>({
    capture_fps: 0,
    capture_ms: null,
    local_detection_ms: null,
    tracking_latency_ms: null,
    tracking_age_ms: null,
    tracking_status: 'waiting',
  });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const controller = useRef<AbortController | null>(null);
  const cloudInFlight = useRef(false);
  const generation = useRef(0);
  const failures = useRef(0);
  const sceneKey = useRef('');
  const localSignature = useRef('');
  const lastCloudAnalysis = useRef(0);
  const lastTrackingAnalysis = useRef(0);
  const trackingInFlight = useRef(false);
  const trackingFailures = useRef(0);
  // Last fresh server tracking answer, lent to local scans until it expires.
  const trackedSnapshot = useRef<TrackedSnapshot | null>(null);
  const withDepth = useCallback((local: VisionResult): VisionResult => (
    { ...local, objects: withTrackedDepth(local.objects, trackedSnapshot.current) }
  ), []);
  const captureTimes = useRef<number[]>([]);
  // Latest result of the browser's live-video detection loop. While it is
  // fresh, the scan loop below uses it instead of capturing and detecting again.
  const liveLocal = useRef<{ result: DetectionResult; at: number } | null>(null);
  const signFrames = useRef<string[]>([]);
  const signTranscript = useRef('');

  const stop = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    controller.current?.abort();
    timer.current = null;
    controller.current = null;
    cloudInFlight.current = false;
    trackingInFlight.current = false;
  }, []);

  const run = useCallback(async (runGeneration: number) => {
    if (!enabled || !camera.current || runGeneration !== generation.current) return;
    setState((current) => current === 'offline' ? 'analyzing' : current);
    let nextDelay = mode === 'sign' ? 700 : LOCAL_SCAN_INTERVAL_MS;
    const activeApiUrl = getApiUrl();
    const live = mode === 'scene' && liveLocal.current && Date.now() - liveLocal.current.at <= LIVE_LOCAL_FRESH_MS
      ? liveLocal.current.result
      : null;
    try {
      if (live) {
        // Boxes already update every video frame; only capture a photo when the
        // server tracker or the cloud description is due for a new frame.
        nextDelay = LIVE_SCAN_INTERVAL_MS;
        const now = Date.now();
        const signature = detectionSignature(live.detections);
        const shouldUseCloud = Boolean(activeApiUrl) && (signature !== localSignature.current || now - lastCloudAnalysis.current >= 8000);
        const trackingDue = Boolean(activeApiUrl) && now - lastTrackingAnalysis.current >= 900 && !trackingInFlight.current;
        if (!trackingDue && !(shouldUseCloud && !cloudInFlight.current)) {
          localSignature.current = signature;
          failures.current = 0;
          setState('live');
          if (!activeApiUrl || !shouldUseCloud) {
            const localResult = withDepth(synthesizeLocalSceneResult(live.detections, sceneKey.current, mode, live.blocked));
            sceneKey.current = localResult.scene_key;
            setResult(localResult);
            setLastUpdated(Date.now());
          }
          return;
        }
      }
      const captureStartedAt = performanceNow();
      const picture = await camera.current.takePictureAsync({ quality: 0.7, skipProcessing: true });
      if (!picture?.uri) throw new Error('Camera did not return an image');
      const capturedAt = Date.now();
      const captureEndedAt = performanceNow();
      captureTimes.current.push(captureEndedAt);
      captureTimes.current = captureTimes.current.filter((stamp) => captureEndedAt - stamp <= 5000);
      const captureFps = captureTimes.current.length > 1
        ? (captureTimes.current.length - 1) * 1000 / (captureTimes.current[captureTimes.current.length - 1] - captureTimes.current[0])
        : 0;
      setPerformance((previous) => ({
        ...previous,
        // While live-video detection runs, it reports its own frame rate.
        capture_fps: live ? previous.capture_fps : Math.round(captureFps * 10) / 10,
        capture_ms: Math.round(captureEndedAt - captureStartedAt),
      }));
      if (mode === 'sign') {
        if (!activeApiUrl) {
          setState('offline');
          return;
        }
        const frame = await imageUriToDataUrl(picture.uri);
        signFrames.current.push(frame);
        if (signFrames.current.length > 3) signFrames.current.shift();
        failures.current = 0;
        setState(signFrames.current.length === 3 ? 'live' : 'analyzing');
        if (signFrames.current.length === 3 && !cloudInFlight.current) {
          const cloudController = new AbortController();
          controller.current = cloudController;
          cloudInFlight.current = true;
          const windowFrames = [...signFrames.current];
          // Retain one boundary frame so consecutive windows overlap without
          // resending an entire window or creating gaps between phrases.
          signFrames.current = [signFrames.current[2]];
          void interpretSignSequence(windowFrames, signTranscript.current, cloudController.signal)
            .then((next) => {
              if (cloudController.signal.aborted || runGeneration !== generation.current) return;
              signTranscript.current = next.transcript;
              setResult({
                summary: next.transcript || 'Watching for an ASL phrase…',
                speech: next.segment || next.transcript || 'No clear sign yet.',
                scene_key: next.transcript,
                urgency: 'low',
                changed: Boolean(next.segment),
                objects: [],
                task: 'sign',
                should_speak: next.should_speak,
              });
              setLastUpdated(Date.now());
            })
            .catch((error) => {
              if (!cloudController.signal.aborted) console.warn('Sign translation unavailable', error);
            })
            .finally(() => {
              if (controller.current === cloudController) controller.current = null;
              cloudInFlight.current = false;
            });
        }
        return;
      }
      let local: DetectionResult;
      if (live) {
        local = live;
      } else {
        const localDetectionStartedAt = performanceNow();
        local = await detectObjects(picture.uri);
        const localDetectionMs = Math.round(performanceNow() - localDetectionStartedAt);
        if (runGeneration !== generation.current) return;
        setDetections(local.detections);
        setBlocked(local.blocked ? { ...local.blocked, at: Date.now() } : null);
        setPerformance((previous) => ({ ...previous, local_detection_ms: localDetectionMs }));
      }
      const signature = detectionSignature(local.detections);
      const now = Date.now();
      const shouldUseCloud = Boolean(activeApiUrl) && (!local.available || signature !== localSignature.current || now - lastCloudAnalysis.current >= 8000);
      localSignature.current = signature;
      failures.current = 0;
      setState('live');

      // Geometry has its own API path and cadence. It can update tracked boxes
      // while Groq is describing an earlier frame; it never creates a spoken
      // collision warning in the client.
      if (activeApiUrl && now - lastTrackingAnalysis.current >= 900 && !trackingInFlight.current) {
        trackingInFlight.current = true;
        const trackingController = new AbortController();
        const trackingFrameStartedAt = performanceNow();
        void trackFrame(picture.uri, trackingController.signal)
          .then((tracking) => {
            if (runGeneration !== generation.current) return;
            const trackingAgeMs = Math.round(performanceNow() - trackingFrameStartedAt);
            lastTrackingAnalysis.current = Date.now();
            trackingFailures.current = 0;
            setPerformance((previous) => ({
              ...previous,
              tracking_latency_ms: tracking.tracking_latency_ms,
              tracking_age_ms: trackingAgeMs,
              tracking_status: trackingAgeMs > 3000 ? 'slow' : 'live',
            }));
            // Do not let a delayed response overwrite newer camera detections.
            if (trackingAgeMs > 3000 || Date.now() - capturedAt > 3000) return;
            setCameraCoach(tracking.camera_coach || null);
            trackedSnapshot.current = { objects: tracking.objects, at: Date.now() };
            const obstaclePoints = (tracking.obstacle_points || []).filter(
              (point): point is [number, number, number] => Array.isArray(point) && point.length >= 3 && point.slice(0, 3).every(Number.isFinite),
            );
            obstacleSequence.current += 1;
            setObstacles({
              sequence: obstacleSequence.current,
              capturedAt,
              points: obstaclePoints,
              rangeM: typeof tracking.obstacle_range_m === 'number' ? tracking.obstacle_range_m : null,
            });
            setResult((previous) => {
              const base = previous || synthesizeLocalSceneResult(local.detections, sceneKey.current, mode, local.blocked);
              return { ...base, objects: tracking.objects };
            });
          })
          .catch((error) => {
            trackingFailures.current += 1;
            const backoffMs = Math.min(10_000, 1000 * (2 ** trackingFailures.current));
            lastTrackingAnalysis.current = Date.now() + backoffMs - 900;
            setPerformance((previous) => ({ ...previous, tracking_status: 'offline' }));
            console.warn(`Experimental server tracking unavailable (retrying in ${Math.round(backoffMs / 1000)}s)`, error);
          })
          .finally(() => { trackingInFlight.current = false; });
      }

      // In purely offline mode (or while waiting for cloud), synthesize local spatial speech
      if (!activeApiUrl || !shouldUseCloud) {
        const localResult = withDepth(synthesizeLocalSceneResult(local.detections, sceneKey.current, mode, local.blocked));
        sceneKey.current = localResult.scene_key;
        setResult(localResult);
        setLastUpdated(Date.now());
      }

      // Never make the bounding boxes wait for a network request. The previous
      // implementation paused local updates while the cloud description ran,
      // leaving frames behind when the user turned the camera.
      if (shouldUseCloud && !cloudInFlight.current) {
        const cloudController = new AbortController();
        controller.current = cloudController;
        cloudInFlight.current = true;
        void analyzeFrame(picture.uri, mode, sceneKey.current, cloudController.signal)
          .then((next) => {
            if (cloudController.signal.aborted || runGeneration !== generation.current) return;
            sceneKey.current = next.scene_key;
            lastCloudAnalysis.current = Date.now();
            setResult(next);
            setLastUpdated(Date.now());
          })
          .catch((error) => {
            if (cloudController.signal.aborted || runGeneration !== generation.current) return;
            // Local detection remains live even when optional cloud narration is unavailable.
            lastCloudAnalysis.current = Date.now() + 10_000;
            console.warn('Cloud scene description unavailable (backing off 10s)', error);
            // Fall back to local synthesis on cloud failure
            const localFallback = withDepth(synthesizeLocalSceneResult(local.detections, sceneKey.current, mode, local.blocked));
            sceneKey.current = localFallback.scene_key;
            setResult(localFallback);
            setLastUpdated(Date.now());
          })
          .finally(() => {
            if (controller.current === cloudController) controller.current = null;
            cloudInFlight.current = false;
          });
      }
    } catch (error) {
      if (runGeneration !== generation.current) return;
      failures.current += 1;
      setState('offline');
      const retryAfter = (error as Error & { retryAfter?: number }).retryAfter || 0;
      nextDelay = Math.max(retryAfter * 1000, Math.min(2000 * 2 ** (failures.current - 1), 15000));
    } finally {
      controller.current = null;
      if (enabled && runGeneration === generation.current) {
        timer.current = setTimeout(() => void run(runGeneration), nextDelay);
      }
    }
  }, [camera, enabled, mode, withDepth]);

  // Browser: run the local detector on every displayed video frame, so the
  // boxes follow the camera at the detector's real speed (about 30 fps or more
  // on WebGPU) instead of the photo-capture cadence.
  useEffect(() => {
    if (!enabled || mode !== 'scene' || typeof document === 'undefined') return;
    let active = true;
    const frameTimes: number[] = [];
    let lastReport = 0;
    void warmUpDetector().catch(() => {});
    const loop = async () => {
      while (active) {
        const video = findCameraVideo();
        const started = performanceNow();
        const pending = video ? detectVideoFrame(video) : null;
        if (!pending) {
          await new Promise((resolve) => setTimeout(resolve, 150));
          continue;
        }
        const local = await pending;
        if (!active) return;
        if (!local.available) {
          liveLocal.current = null;
          await new Promise((resolve) => setTimeout(resolve, 1000));
          continue;
        }
        const finished = performanceNow();
        liveLocal.current = { result: local, at: Date.now() };
        setDetections(local.detections);
        setBlocked(local.blocked ? { ...local.blocked, at: Date.now() } : null);
        frameTimes.push(finished);
        while (frameTimes.length && finished - frameTimes[0] > 2000) frameTimes.shift();
        if (finished - lastReport >= 500) {
          lastReport = finished;
          const fps = frameTimes.length > 1 ? (frameTimes.length - 1) * 1000 / (finished - frameTimes[0]) : 0;
          setPerformance((previous) => ({
            ...previous,
            capture_fps: Math.round(fps * 10) / 10,
            local_detection_ms: Math.round(finished - started),
            local_backend: localDetectorBackend(),
          }));
        }
        // Yield to the browser so the new boxes paint before the next frame.
        // CPU inference blocks the page while it runs, so leave it some room.
        if (localDetectorBackend() === 'wasm') await new Promise((resolve) => setTimeout(resolve, 120));
        await nextAnimationFrame();
      }
    };
    void loop();
    return () => {
      active = false;
      liveLocal.current = null;
    };
  }, [enabled, mode]);

  const scanNow = useCallback(() => {
    stop();
    const current = generation.current;
    void run(current);
  }, [run, stop]);

  useEffect(() => {
    stop();
    generation.current += 1;
    sceneKey.current = '';
    localSignature.current = '';
    lastCloudAnalysis.current = 0;
    lastTrackingAnalysis.current = 0;
    trackingInFlight.current = false;
    trackingFailures.current = 0;
    captureTimes.current = [];
    signFrames.current = [];
    signTranscript.current = '';
    failures.current = 0;
    setResult(null);
    setCameraCoach(null);
    trackedSnapshot.current = null;
    setObstacles(null);
    setBlocked(null);
    setPerformance({ capture_fps: 0, capture_ms: null, local_detection_ms: null, tracking_latency_ms: null, tracking_age_ms: null, tracking_status: 'waiting' });
    if (!enabled) {
      setState('paused');
      return stop;
    }
    const current = generation.current;
    timer.current = setTimeout(() => void run(current), 350);
    return stop;
  }, [enabled, mode, run, stop]);

  return { result, state, lastUpdated, detections, cameraCoach, obstacles, blocked, performance, scanNow };
}

function performanceNow() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

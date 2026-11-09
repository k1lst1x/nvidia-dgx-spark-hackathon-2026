import type { CameraView } from 'expo-camera';
import { RefObject, useCallback, useEffect, useRef, useState } from 'react';
import { detectObjects } from './detector';
import type { LocalDetection } from './detector.types';

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
};

export type ApproachingWarning = { label: string; distance_m: number };

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
  experimental: true;
  collision_warnings_enabled: false;
};

export type SignTranslation = {
  transcript: string;
  segment: string;
  confidence: number;
  is_final: boolean;
  should_speak: boolean;
};

type VisionState = 'starting' | 'analyzing' | 'live' | 'offline' | 'paused';

// Local object frames should follow camera movement; cloud narration can update less often.
const LOCAL_SCAN_INTERVAL_MS = 200;

export function getApiUrl(): string {
  if (typeof window !== 'undefined' && window.localStorage) {
    const custom = window.localStorage.getItem('guidelens_api_url');
    if (custom && custom.trim()) return custom.trim().replace(/\/$/, '');
  }
  const configured = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
  if (configured) return configured;
  if (typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname)) {
    return 'http://localhost:8000';
  }
  return '';
}

const configuredApiUrl = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
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

export function synthesizeLocalSceneResult(
  detections: LocalDetection[],
  previousSceneKey: string,
  mode: AssistMode,
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
    const scene_key = 'empty_path';
    return {
      summary: 'Path clear ahead',
      speech: 'The path is clear. No obstacles detected nearby.',
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
  const signFrames = useRef<string[]>([]);
  const signTranscript = useRef('');

  const stop = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    controller.current?.abort();
    timer.current = null;
    controller.current = null;
    cloudInFlight.current = false;
  }, []);

  const run = useCallback(async (runGeneration: number) => {
    if (!enabled || !camera.current || runGeneration !== generation.current) return;
    setState((current) => current === 'offline' ? 'analyzing' : current);
    let nextDelay = mode === 'sign' ? 700 : LOCAL_SCAN_INTERVAL_MS;
    const activeApiUrl = getApiUrl();
    try {
      const picture = await camera.current.takePictureAsync({ quality: 0.35, skipProcessing: true });
      if (!picture?.uri) throw new Error('Camera did not return an image');
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
      const local = await detectObjects(picture.uri);
      if (runGeneration !== generation.current) return;
      setDetections(local.detections);
      const signature = local.detections
        .filter((item) => item.confidence >= 0.5)
        .map((item) => item.label)
        .sort()
        .join('|');
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
        void trackFrame(picture.uri, trackingController.signal)
          .then((tracking) => {
            if (runGeneration !== generation.current) return;
            lastTrackingAnalysis.current = Date.now();
            setResult((previous) => {
              const base = previous || synthesizeLocalSceneResult(local.detections, sceneKey.current, mode);
              return { ...base, objects: tracking.objects };
            });
          })
          .catch((error) => console.warn('Experimental server tracking unavailable', error))
          .finally(() => { trackingInFlight.current = false; });
      }

      // In purely offline mode (or while waiting for cloud), synthesize local spatial speech
      if (!activeApiUrl || !shouldUseCloud) {
        const localResult = synthesizeLocalSceneResult(local.detections, sceneKey.current, mode);
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
            console.warn('Cloud scene description unavailable', error);
            // Fall back to local synthesis on cloud failure
            const localFallback = synthesizeLocalSceneResult(local.detections, sceneKey.current, mode);
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
  }, [camera, enabled, mode]);

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
    signFrames.current = [];
    signTranscript.current = '';
    failures.current = 0;
    setResult(null);
    if (!enabled) {
      setState('paused');
      return stop;
    }
    const current = generation.current;
    timer.current = setTimeout(() => void run(current), 350);
    return stop;
  }, [enabled, mode, run, stop]);

  return { result, state, lastUpdated, detections, scanNow };
}

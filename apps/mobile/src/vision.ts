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

const configuredApiUrl = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
const API_URL = configuredApiUrl || (
  typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname)
    ? 'http://localhost:8000'
    : ''
);
export const hasVisionBackend = Boolean(API_URL);
const sessionId = `gl-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

function requireApiUrl() {
  if (!API_URL) throw new Error('EXPO_PUBLIC_API_URL is not configured');
  return API_URL;
}

async function analyzeFrame(uri: string, mode: AssistMode, previousSceneKey: string, signal: AbortSignal) {
  const apiUrl = requireApiUrl();
  const data = new FormData();
  // Browsers require a real Blob in multipart form data; the React Native
  // `{ uri, name, type }` object otherwise becomes a text field and FastAPI
  // returns 422 before the vision model ever sees the image.
  if (typeof window !== 'undefined') {
    const imageResponse = await fetch(uri);
    if (!imageResponse.ok) throw new Error('Camera frame could not be prepared for analysis');
    const image = await imageResponse.blob();
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
    throw new Error(`Vision server is unavailable at ${apiUrl}. Start the API server and try again.`);
  }
  if (!response.ok) {
    const retryAfter = Number(response.headers.get('retry-after') || 0);
    const error = new Error(`Vision request failed: ${response.status}`) as Error & { retryAfter?: number };
    error.retryAfter = retryAfter;
    throw error;
  }
  return (await response.json()) as VisionResult;
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
    throw new Error(`Translation server is unavailable at ${apiUrl}. Start the API server and try again.`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail || `Sign interpretation failed: ${response.status}`);
  }
  return (await response.json()) as SignTranslation;
}

async function imageUriToDataUrl(uri: string) {
  if (uri.startsWith('data:image/jpeg;base64,')) return uri;
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
    distance_m: d.distance_m ?? null,
    relative_proximity: d.distance_m != null ? Math.max(0, Math.min(1, 1 - d.distance_m / 4.0)) : null,
    track_id: d.id ?? null,
  }));

  // 1. Check for approaching hazards (TTC alert)
  const approachingHazards = detections.filter(
    (d) => d.is_approaching && d.distance_m != null && d.distance_m <= 2.5
  );

  if (approachingHazards.length > 0) {
    const hazard = approachingHazards[0];
    const distStr = hazard.distance_m ? `${hazard.distance_m.toFixed(1)} meters` : 'close';
    const speech = `Caution: ${hazard.label} approaching directly ahead at ${distStr}.`;
    const scene_key = `hazard_${hazard.label}_${Math.round((hazard.distance_m || 1) * 2)}`;
    return {
      summary: `Caution: ${hazard.label} approaching (~${distStr})`,
      speech,
      scene_key,
      urgency: 'high',
      changed: true,
      objects: visionObjects,
      task: mode,
      should_speak: true,
      approaching: approachingHazards.map((h) => ({
        label: h.label,
        distance_m: h.distance_m ?? 1.0,
      })),
    };
  }

  // 2. Check for close obstacles directly in path (< 1.4m)
  const immediateAhead = detections.filter(
    (d) => d.direction === 'ahead' && d.distance_m != null && d.distance_m < 1.4
  );

  if (immediateAhead.length > 0) {
    const obs = immediateAhead[0];
    const speech = `Attention: ${obs.label} directly ahead, ${obs.distance_m?.toFixed(1)} meters.`;
    const scene_key = `obs_${obs.label}_${Math.round((obs.distance_m || 1) * 2)}`;
    return {
      summary: `Obstacle ahead: ${obs.label} (${obs.distance_m?.toFixed(1)}m)`,
      speech,
      scene_key,
      urgency: 'medium',
      changed: scene_key !== previousSceneKey,
      objects: visionObjects,
      task: mode,
      should_speak: scene_key !== previousSceneKey,
    };
  }

  // 3. Clear path
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

  // 4. Natural spatial narration of detected objects
  const ahead = detections.filter((d) => d.direction === 'ahead');
  const left = detections.filter((d) => d.direction === 'left');
  const right = detections.filter((d) => d.direction === 'right');

  const parts: string[] = [];
  if (ahead.length > 0) {
    parts.push(`Ahead: ${ahead.slice(0, 2).map((d) => `${d.label} at ${d.distance_m?.toFixed(1) || '1'}m`).join(', ')}`);
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
    .map((d) => `${d.label}_${d.direction}_${Math.round(d.distance_m || 1)}`)
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
    try {
      const picture = await camera.current.takePictureAsync({ quality: 0.35, skipProcessing: true });
      if (!picture?.uri) throw new Error('Camera did not return an image');
      if (mode === 'sign') {
        if (!API_URL) {
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
      const shouldUseCloud = Boolean(API_URL) && (!local.available || signature !== localSignature.current || now - lastCloudAnalysis.current >= 8000);
      localSignature.current = signature;
      failures.current = 0;
      setState('live');

      // In purely offline mode (or while waiting for cloud), synthesize local spatial speech
      if (!API_URL || !shouldUseCloud) {
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

import type { CameraView } from 'expo-camera';
import { RefObject, useCallback, useEffect, useRef, useState } from 'react';
import { detectObjects } from './detector';
import type { LocalDetection } from './detector.types';

export type AssistMode = 'scene' | 'sign';

export type VisionObject = {
  label: string;
  confidence: number;
  position: string;
};

export type VisionResult = {
  summary: string;
  speech: string;
  scene_key: string;
  urgency: 'low' | 'medium' | 'high';
  changed: boolean;
  objects: VisionObject[];
  task: AssistMode;
};

type VisionState = 'starting' | 'analyzing' | 'live' | 'offline' | 'paused';

const API_URL = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');
const sessionId = `gl-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

async function analyzeFrame(uri: string, mode: AssistMode, previousSceneKey: string, signal: AbortSignal) {
  if (!API_URL) throw new Error('EXPO_PUBLIC_API_URL is not configured');
  const data = new FormData();
  data.append('image', { uri, name: 'frame.jpg', type: 'image/jpeg' } as unknown as Blob);
  data.append('task', mode);
  data.append('previous_scene_key', previousSceneKey);
  const response = await fetch(`${API_URL}/v1/describe`, {
    method: 'POST',
    body: data,
    headers: { 'X-Session-ID': sessionId },
    signal,
  });
  if (!response.ok) {
    const retryAfter = Number(response.headers.get('retry-after') || 0);
    const error = new Error(`Vision request failed: ${response.status}`) as Error & { retryAfter?: number };
    error.retryAfter = retryAfter;
    throw error;
  }
  return (await response.json()) as VisionResult;
}

export function useVisionLoop(camera: RefObject<CameraView | null>, mode: AssistMode, enabled: boolean) {
  const [result, setResult] = useState<VisionResult | null>(null);
  const [state, setState] = useState<VisionState>('starting');
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [detections, setDetections] = useState<LocalDetection[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const controller = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const failures = useRef(0);
  const sceneKey = useRef('');
  const localSignature = useRef('');
  const lastCloudAnalysis = useRef(0);

  const stop = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    controller.current?.abort();
    timer.current = null;
    controller.current = null;
  }, []);

  const run = useCallback(async (runGeneration: number) => {
    if (!enabled || !camera.current || runGeneration !== generation.current) return;
    setState('analyzing');
    controller.current = new AbortController();
    let nextDelay = 2800;
    try {
      const picture = await camera.current.takePictureAsync({ quality: 0.35, skipProcessing: true });
      if (!picture?.uri) throw new Error('Camera did not return an image');
      const local = await detectObjects(picture.uri);
      setDetections(local.detections);
      const signature = local.detections
        .filter((item) => item.confidence >= 0.5)
        .map((item) => item.label)
        .sort()
        .join('|');
      const now = Date.now();
      const shouldUseCloud = !local.available || signature !== localSignature.current || now - lastCloudAnalysis.current >= 8000;
      localSignature.current = signature;
      if (!shouldUseCloud) {
        failures.current = 0;
        setState('live');
        nextDelay = 900;
        return;
      }
      const next = await analyzeFrame(picture.uri, mode, sceneKey.current, controller.current.signal);
      if (runGeneration !== generation.current) return;
      failures.current = 0;
      sceneKey.current = next.scene_key;
      lastCloudAnalysis.current = Date.now();
      setResult(next);
      setLastUpdated(Date.now());
      setState('live');
      nextDelay = next.urgency === 'high' ? 1400 : next.changed ? 2400 : 4500;
    } catch (error) {
      if (controller.current?.signal.aborted || runGeneration !== generation.current) return;
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

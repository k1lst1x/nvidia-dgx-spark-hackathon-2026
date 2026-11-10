import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import * as THREE from 'three';
import type { VisionObject, VisionPerformance } from '../src/vision';

type MapDetection = Pick<VisionObject, 'label' | 'confidence' | 'x' | 'y' | 'width' | 'height' | 'distance_m' | 'track_id' | 'spatial'>;
type Props = { detections: MapDetection[]; performance: VisionPerformance };
type Anchor = { label: string; x: number; y: number; z: number; lastSeenAt: number };
type MapStats = { objects: number; depthObjects: number; mapHz: number; renderFps: number };

const EYE_HEIGHT_M = 1.45;
const MAP_INTERVAL_MS = 250;

function labelSprite(text: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) return new THREE.Sprite();
  context.fillStyle = 'rgba(8, 16, 12, 0.92)';
  context.fillRect(6, 8, 244, 48);
  context.strokeStyle = '#B7FF56';
  context.strokeRect(6, 8, 244, 48);
  context.fillStyle = '#F4FFE4';
  context.font = '700 26px sans-serif';
  context.fillText(text.slice(0, 16), 16, 42);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false }));
  sprite.scale.set(0.95, 0.24, 1);
  return sprite;
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    (Array.isArray(material) ? material : material ? [material] : []).forEach((item) => {
      (item as THREE.SpriteMaterial).map?.dispose?.();
      item.dispose();
    });
  });
}

function performanceNow() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export default function LiveSceneMap({ detections, performance }: Props) {
  const host = useRef<HTMLElement | null>(null);
  const detectionsRef = useRef(detections);
  const performanceRef = useRef(performance);
  const resetRef = useRef<() => void>(() => {});
  const [stats, setStats] = useState<MapStats>({ objects: 0, depthObjects: 0, mapHz: 0, renderFps: 0 });
  detectionsRef.current = detections;
  performanceRef.current = performance;

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07110c);
    scene.fog = new THREE.Fog(0x07110c, 8, 18);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 40);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    container.appendChild(renderer.domElement);

    const grid = new THREE.GridHelper(10, 10, 0x1d3a2c, 0x143024);
    scene.add(grid);
    const cameraMarker = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.22, 4), new THREE.MeshBasicMaterial({ color: 0xb7ff56, wireframe: true }));
    cameraMarker.position.set(0, EYE_HEIGHT_M, 0);
    cameraMarker.rotation.x = -Math.PI / 2;
    scene.add(cameraMarker);
    const objectGroup = new THREE.Group();
    scene.add(objectGroup);
    const anchors = new Map<string, Anchor>();
    const anchorMeshes = new Map<string, { marker: THREE.Mesh; label: THREE.Sprite }>();
    let integrationCount = 0;
    let integrationStarted = performanceNow();
    let measuredMapHz = 0;
    let renderedFrames = 0;
    let renderStarted = performanceNow();

    const syncObjects = () => {
      anchors.forEach((anchor, key) => {
        let visual = anchorMeshes.get(key);
        if (!visual) {
          visual = {
            marker: new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.28, 0.28), new THREE.MeshBasicMaterial({ color: 0xb7ff56, wireframe: true })),
            label: labelSprite(anchor.label),
          };
          anchorMeshes.set(key, visual);
          objectGroup.add(visual.marker, visual.label);
        }
        visual.marker.position.set(anchor.x, anchor.y, anchor.z);
        visual.label.position.set(anchor.x, anchor.y + 0.34, anchor.z);
      });
      anchorMeshes.forEach((visual, key) => {
        if (anchors.has(key)) return;
        objectGroup.remove(visual.marker, visual.label);
        disposeObject(visual.marker);
        disposeObject(visual.label);
        anchorMeshes.delete(key);
      });
    };

    const integrate = () => {
      const now = Date.now();
      const seen = detectionsRef.current;
      let depthObjects = 0;
      seen.slice(0, 8).forEach((item, index) => {
        const center = item.spatial?.center_m;
        const validCenter = Array.isArray(center) && center.length >= 3 && center.slice(0, 3).every(Number.isFinite);
        const depth = typeof item.distance_m === 'number' && Number.isFinite(item.distance_m) && item.distance_m > 0 ? item.distance_m : null;
        // Do not invent a default 2–3m range. Show only observations with an actual depth result.
        if (!validCenter && depth == null) return;
        depthObjects += 1;
        const nx = ((item.x ?? 0.5) + (item.width ?? 0) / 2 - 0.5) * 2;
        const ny = (0.5 - ((item.y ?? 0.5) + (item.height ?? 0) / 2)) * 2;
        const point = validCenter
          ? new THREE.Vector3(center![0], EYE_HEIGHT_M - center![1], -Math.abs(center![2]))
          : new THREE.Vector3(nx * depth! * 0.55, EYE_HEIGHT_M + ny * depth! * 0.42, -depth!);
        const key = String(item.track_id ?? `${item.label}-${index}`);
        const previous = anchors.get(key);
        if (previous) {
          previous.x = previous.x * 0.35 + point.x * 0.65;
          previous.y = previous.y * 0.35 + point.y * 0.65;
          previous.z = previous.z * 0.35 + point.z * 0.65;
          previous.label = item.label;
          previous.lastSeenAt = now;
        } else {
          anchors.set(key, { label: item.label, x: point.x, y: point.y, z: point.z, lastSeenAt: now });
        }
      });
      anchors.forEach((anchor, key) => { if (now - anchor.lastSeenAt > 1800) anchors.delete(key); });
      syncObjects();
      integrationCount += 1;
      const elapsed = performanceNow() - integrationStarted;
      if (elapsed >= 1000) {
        measuredMapHz = Math.round(integrationCount * 1000 / elapsed * 10) / 10;
        integrationCount = 0;
        integrationStarted = performanceNow();
      }
      setStats((previous) => ({ ...previous, objects: anchors.size, depthObjects, mapHz: measuredMapHz }));
    };
    const resize = () => {
      const width = Math.max(container.clientWidth, 220);
      const height = Math.max(container.clientHeight, 140);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    camera.position.set(0, 2.15, 2.4);
    camera.lookAt(0, 0.9, -1.6);
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    const timer = window.setInterval(integrate, MAP_INTERVAL_MS);
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      renderedFrames += 1;
      const now = performanceNow();
      if (now - renderStarted >= 1000) {
        const renderFps = Math.round(renderedFrames * 1000 / (now - renderStarted));
        renderedFrames = 0;
        renderStarted = now;
        setStats((previous) => ({ ...previous, renderFps }));
      }
      renderer.render(scene, camera);
    };
    tick();
    integrate();

    resetRef.current = () => {
      anchors.clear();
      anchorMeshes.clear();
      disposeObject(objectGroup);
      objectGroup.clear();
      setStats({ objects: 0, depthObjects: 0, mapHz: 0, renderFps: 0 });
    };

    return () => {
      window.clearInterval(timer);
      cancelAnimationFrame(frame);
      observer.disconnect();
      disposeObject(objectGroup);
      cameraMarker.geometry.dispose();
      (cameraMarker.material as THREE.Material).dispose();
      grid.geometry.dispose();
      const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
      gridMaterials.forEach((material) => material.dispose());
      renderer.dispose();
      container.replaceChildren();
    };
  }, []);

  return (
    <View style={styles.panel}>
      <View style={styles.header}>
        <Text style={styles.kicker}>LIVE 3D · RELATIVE</Text>
        <Text style={styles.count}>{stats.objects} objects</Text>
        <Pressable onPress={() => resetRef.current()} style={styles.clear}><Text style={styles.clearText}>Clear</Text></Pressable>
      </View>
      <View ref={(node) => {
        const candidate = node as unknown as HTMLElement | { _nativeNode?: HTMLElement } | null;
        host.current = candidate && 'appendChild' in candidate ? candidate : candidate?._nativeNode ?? null;
      }} style={styles.canvas} />
      <Text style={styles.note}>
        Scan {performance.capture_fps.toFixed(1)} fps · local {performance.local_detection_ms ?? '—'} ms · API {performance.tracking_latency_ms ?? '—'} ms / {performance.tracking_age_ms ?? '—'} ms ({performance.tracking_status})
        {'\n'}Map {stats.mapHz.toFixed(1)} Hz · render {stats.renderFps} fps · depth anchors {stats.depthObjects}
        {'\n'}Camera-relative preview; not persistent SLAM. Objects without depth are omitted.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  panel: { position: 'absolute', right: 0, top: 0, width: 300, backgroundColor: 'rgba(5, 12, 9, 0.9)', borderColor: 'rgba(183, 255, 86, 0.45)', borderWidth: 1, borderRadius: 16, padding: 8, gap: 6, zIndex: 6 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  kicker: { color: '#B7FF56', fontSize: 10, fontWeight: '900', letterSpacing: 0.8 },
  count: { color: '#D5E6D8', fontSize: 10, fontWeight: '700', flex: 1 },
  clear: { backgroundColor: '#163024', borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  clearText: { color: '#E7F6EA', fontSize: 10, fontWeight: '800' },
  canvas: { height: 168, borderRadius: 10, overflow: 'hidden', backgroundColor: '#07110C' },
  note: { color: '#B7C9BC', fontSize: 9, lineHeight: 12 },
});

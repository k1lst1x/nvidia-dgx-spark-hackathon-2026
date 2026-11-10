import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import * as THREE from 'three';
import type { ObstacleFrame, VisionObject, VisionPerformance } from '../src/vision';

type MapDetection = Pick<VisionObject, 'label' | 'confidence' | 'x' | 'y' | 'width' | 'height' | 'distance_m' | 'track_id' | 'spatial'>;
type Props = { detections: MapDetection[]; obstacles?: ObstacleFrame | null; performance: VisionPerformance };
type Anchor = { label: string; x: number; y: number; z: number; firstSeenAt: number; lastSeenAt: number; seenTick: number };
type Dot = { x: number; y: number; z: number; seenAt: number; hits: number };
type MapStats = { liveObjects: number; rememberedObjects: number; depthObjects: number; dots: number; mapHz: number; renderFps: number };

const EYE_HEIGHT_M = 1.45;
const MAP_INTERVAL_MS = 250;

// Object memory: a detected object stays on the map after it leaves view,
// fading out, then is forgotten. The cap keeps the scene small.
const OBJECT_FRESH_MS = 2000;
const OBJECT_MEMORY_MS = 15_000;
const MAX_OBJECT_ANCHORS = 24;
// Untracked detections (or a tracker that re-issued an ID) are merged with a
// remembered anchor of the same label within this radius.
const OBJECT_MATCH_RADIUS_M = 0.8;

// Obstacle dots: unlabeled surfaces (walls, trees, poles) from metric depth.
// Only near surfaces are kept, each is forgotten after a few seconds, dots are
// merged on a voxel grid, and the total is hard-capped.
const OBSTACLE_RANGE_M = 4.5;
const OBSTACLE_MEMORY_MS = 8000;
const OBSTACLE_VOXEL_M = 0.15;
const MAX_OBSTACLE_DOTS = 1200;

const BACKGROUND = new THREE.Color(0x07110c);
const NEAR_COLOR = new THREE.Color(0xff6b4a);
const FAR_COLOR = new THREE.Color(0x56c8ff);
const LIVE_OBJECT_COLOR = new THREE.Color(0xb7ff56);
const REMEMBERED_OBJECT_COLOR = new THREE.Color(0x6f8f78);

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

function voxelKey(x: number, y: number, z: number) {
  return `${Math.round(x / OBSTACLE_VOXEL_M)}:${Math.round(y / OBSTACLE_VOXEL_M)}:${Math.round(z / OBSTACLE_VOXEL_M)}`;
}

/** Map a camera-frame point (x right, y down, z forward) into the scene. */
function toScene(x: number, y: number, z: number) {
  return { x, y: EYE_HEIGHT_M - y, z: -Math.abs(z) };
}

function distanceFromCamera(point: { x: number; y: number; z: number }) {
  return Math.hypot(point.x, point.y - EYE_HEIGHT_M, point.z);
}

export default function LiveSceneMap({ detections, obstacles = null, performance }: Props) {
  const host = useRef<HTMLElement | null>(null);
  const detectionsRef = useRef(detections);
  const obstaclesRef = useRef(obstacles);
  const performanceRef = useRef(performance);
  const resetRef = useRef<() => void>(() => {});
  const [stats, setStats] = useState<MapStats>({ liveObjects: 0, rememberedObjects: 0, depthObjects: 0, dots: 0, mapHz: 0, renderFps: 0 });
  detectionsRef.current = detections;
  obstaclesRef.current = obstacles;
  performanceRef.current = performance;

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = BACKGROUND.clone();
    scene.fog = new THREE.Fog(0x07110c, 8, 18);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 40);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    container.appendChild(renderer.domElement);

    const grid = new THREE.GridHelper(10, 10, 0x1d3a2c, 0x143024);
    scene.add(grid);
    // Ring on the floor marking how far obstacle dots are projected.
    const rangeRing = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(
        Array.from({ length: 64 }, (_, index) => {
          const angle = (index / 64) * Math.PI * 2;
          return new THREE.Vector3(Math.cos(angle) * OBSTACLE_RANGE_M, 0.01, Math.sin(angle) * OBSTACLE_RANGE_M);
        }),
      ),
      new THREE.LineBasicMaterial({ color: 0x2e5a44 }),
    );
    scene.add(rangeRing);
    const cameraMarker = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.22, 4), new THREE.MeshBasicMaterial({ color: 0xb7ff56, wireframe: true }));
    cameraMarker.position.set(0, EYE_HEIGHT_M, 0);
    cameraMarker.rotation.x = -Math.PI / 2;
    scene.add(cameraMarker);
    const objectGroup = new THREE.Group();
    scene.add(objectGroup);

    // One preallocated point buffer for every obstacle dot: no per-dot meshes,
    // no allocation per frame, and a fixed upper bound on GPU work.
    const dotPositions = new Float32Array(MAX_OBSTACLE_DOTS * 3);
    const dotColors = new Float32Array(MAX_OBSTACLE_DOTS * 3);
    const dotGeometry = new THREE.BufferGeometry();
    const positionAttribute = new THREE.BufferAttribute(dotPositions, 3).setUsage(THREE.DynamicDrawUsage);
    const colorAttribute = new THREE.BufferAttribute(dotColors, 3).setUsage(THREE.DynamicDrawUsage);
    dotGeometry.setAttribute('position', positionAttribute);
    dotGeometry.setAttribute('color', colorAttribute);
    dotGeometry.setDrawRange(0, 0);
    const dotMaterial = new THREE.PointsMaterial({ size: 0.07, vertexColors: true, sizeAttenuation: true, depthWrite: false });
    const dotCloud = new THREE.Points(dotGeometry, dotMaterial);
    dotCloud.frustumCulled = false;
    scene.add(dotCloud);

    const anchors = new Map<string, Anchor>();
    const anchorMeshes = new Map<string, { marker: THREE.Mesh; label: THREE.Sprite }>();
    // Insertion order doubles as age order: a refreshed dot is re-inserted at the end.
    const dots = new Map<string, Dot>();
    let lastObstacleSequence = -1;
    let anchorCounter = 0;
    let tickCounter = 0;
    let integrationCount = 0;
    let integrationStarted = performanceNow();
    let measuredMapHz = 0;
    let renderedFrames = 0;
    let renderStarted = performanceNow();
    const fadeColor = new THREE.Color();

    const syncObjects = (now: number) => {
      anchors.forEach((anchor, key) => {
        let visual = anchorMeshes.get(key);
        if (!visual) {
          visual = {
            marker: new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.28, 0.28), new THREE.MeshBasicMaterial({ color: 0xb7ff56, wireframe: true, transparent: true })),
            label: labelSprite(anchor.label),
          };
          anchorMeshes.set(key, visual);
          objectGroup.add(visual.marker, visual.label);
        }
        const age = now - anchor.lastSeenAt;
        const live = age <= OBJECT_FRESH_MS;
        // Remembered objects dim from 70% to 15% over their memory window.
        const memoryProgress = live ? 0 : Math.min(1, (age - OBJECT_FRESH_MS) / (OBJECT_MEMORY_MS - OBJECT_FRESH_MS));
        const opacity = live ? 1 : 0.7 - memoryProgress * 0.55;
        const markerMaterial = visual.marker.material as THREE.MeshBasicMaterial;
        markerMaterial.color.copy(live ? LIVE_OBJECT_COLOR : REMEMBERED_OBJECT_COLOR);
        markerMaterial.opacity = opacity;
        (visual.label.material as THREE.SpriteMaterial).opacity = opacity;
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

    /** Find a remembered anchor this detection belongs to, if any. */
    const matchAnchor = (key: string | null, label: string, point: THREE.Vector3): string | null => {
      if (key && anchors.has(key)) return key;
      let best: string | null = null;
      let bestDistance = OBJECT_MATCH_RADIUS_M;
      for (const [candidate, anchor] of anchors) {
        // Never merge two objects that are both visible in this frame.
        if (anchor.label !== label || anchor.seenTick === tickCounter) continue;
        const distance = Math.hypot(anchor.x - point.x, anchor.y - point.y, anchor.z - point.z);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = candidate;
        }
      }
      return best;
    };

    const integrateObjects = (now: number) => {
      let depthObjects = 0;
      detectionsRef.current.slice(0, 8).forEach((item) => {
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
        const trackKey = item.track_id != null ? `track:${item.track_id}` : null;
        const matched = matchAnchor(trackKey, item.label, point);
        const previous = matched ? anchors.get(matched) : undefined;
        if (previous && matched) {
          previous.x = previous.x * 0.35 + point.x * 0.65;
          previous.y = previous.y * 0.35 + point.y * 0.65;
          previous.z = previous.z * 0.35 + point.z * 0.65;
          previous.label = item.label;
          previous.lastSeenAt = now;
          previous.seenTick = tickCounter;
          // A tracker that re-issued an ID takes over the remembered anchor.
          if (trackKey && matched !== trackKey) {
            anchors.delete(matched);
            anchors.set(trackKey, previous);
            const visual = anchorMeshes.get(matched);
            if (visual) {
              anchorMeshes.delete(matched);
              anchorMeshes.set(trackKey, visual);
            }
          }
        } else {
          anchorCounter += 1;
          anchors.set(trackKey ?? `seen:${item.label}:${anchorCounter}`, {
            label: item.label, x: point.x, y: point.y, z: point.z, firstSeenAt: now, lastSeenAt: now, seenTick: tickCounter,
          });
        }
      });
      anchors.forEach((anchor, key) => { if (now - anchor.lastSeenAt > OBJECT_MEMORY_MS) anchors.delete(key); });
      if (anchors.size > MAX_OBJECT_ANCHORS) {
        [...anchors.entries()]
          .sort((left, right) => left[1].lastSeenAt - right[1].lastSeenAt)
          .slice(0, anchors.size - MAX_OBJECT_ANCHORS)
          .forEach(([key]) => anchors.delete(key));
      }
      return depthObjects;
    };

    const integrateObstacles = (now: number) => {
      const frame = obstaclesRef.current;
      if (frame && frame.sequence !== lastObstacleSequence) {
        lastObstacleSequence = frame.sequence;
        const range = Math.min(OBSTACLE_RANGE_M, frame.rangeM ?? OBSTACLE_RANGE_M);
        frame.points.forEach(([x, y, z]) => {
          const point = toScene(x, y, z);
          if (distanceFromCamera(point) > range) return;
          const key = voxelKey(point.x, point.y, point.z);
          const previous = dots.get(key);
          if (previous) {
            // Move to the end so the map stays ordered oldest → newest.
            dots.delete(key);
            previous.x = previous.x * 0.5 + point.x * 0.5;
            previous.y = previous.y * 0.5 + point.y * 0.5;
            previous.z = previous.z * 0.5 + point.z * 0.5;
            previous.seenAt = now;
            previous.hits += 1;
            dots.set(key, previous);
          } else {
            dots.set(key, { ...point, seenAt: now, hits: 1 });
          }
        });
      }
      // Oldest first: stop at the first dot that is still inside its window.
      for (const [key, dot] of dots) {
        if (now - dot.seenAt <= OBSTACLE_MEMORY_MS && dots.size <= MAX_OBSTACLE_DOTS) break;
        dots.delete(key);
      }
      let index = 0;
      dots.forEach((dot) => {
        const offset = index * 3;
        dotPositions[offset] = dot.x;
        dotPositions[offset + 1] = dot.y;
        dotPositions[offset + 2] = dot.z;
        // Red when close, blue toward the range limit; fades into the background as it ages.
        const nearness = 1 - Math.min(1, distanceFromCamera(dot) / OBSTACLE_RANGE_M);
        const freshness = 1 - Math.min(1, (now - dot.seenAt) / OBSTACLE_MEMORY_MS);
        fadeColor.copy(FAR_COLOR).lerp(NEAR_COLOR, nearness).lerp(BACKGROUND, (1 - freshness) * 0.85);
        dotColors[offset] = fadeColor.r;
        dotColors[offset + 1] = fadeColor.g;
        dotColors[offset + 2] = fadeColor.b;
        index += 1;
      });
      dotGeometry.setDrawRange(0, index);
      positionAttribute.needsUpdate = true;
      colorAttribute.needsUpdate = true;
      return index;
    };

    const integrate = () => {
      const now = Date.now();
      tickCounter += 1;
      const depthObjects = integrateObjects(now);
      const dotCount = integrateObstacles(now);
      syncObjects(now);
      let liveObjects = 0;
      anchors.forEach((anchor) => { if (now - anchor.lastSeenAt <= OBJECT_FRESH_MS) liveObjects += 1; });
      integrationCount += 1;
      const elapsed = performanceNow() - integrationStarted;
      if (elapsed >= 1000) {
        measuredMapHz = Math.round(integrationCount * 1000 / elapsed * 10) / 10;
        integrationCount = 0;
        integrationStarted = performanceNow();
      }
      setStats((previous) => ({
        ...previous,
        liveObjects,
        rememberedObjects: anchors.size - liveObjects,
        depthObjects,
        dots: dotCount,
        mapHz: measuredMapHz,
      }));
    };
    const resize = () => {
      const width = Math.max(container.clientWidth, 220);
      const height = Math.max(container.clientHeight, 140);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    // Pulled back far enough to frame the whole obstacle range ahead.
    camera.position.set(0, 3.1, 2.6);
    camera.lookAt(0, 0.7, -2.2);
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
      dots.clear();
      // Do not re-ingest the frame that was on screen when Clear was pressed.
      lastObstacleSequence = obstaclesRef.current?.sequence ?? lastObstacleSequence;
      dotGeometry.setDrawRange(0, 0);
      setStats({ liveObjects: 0, rememberedObjects: 0, depthObjects: 0, dots: 0, mapHz: 0, renderFps: 0 });
    };

    return () => {
      window.clearInterval(timer);
      cancelAnimationFrame(frame);
      observer.disconnect();
      disposeObject(objectGroup);
      dotGeometry.dispose();
      dotMaterial.dispose();
      rangeRing.geometry.dispose();
      (rangeRing.material as THREE.Material).dispose();
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
        <Text style={styles.count}>
          {stats.liveObjects} live · {stats.rememberedObjects} remembered · {stats.dots} dots
        </Text>
        <Pressable onPress={() => resetRef.current()} style={styles.clear}><Text style={styles.clearText}>Clear</Text></Pressable>
      </View>
      <View ref={(node) => {
        const candidate = node as unknown as HTMLElement | { _nativeNode?: HTMLElement } | null;
        host.current = candidate && 'appendChild' in candidate ? candidate : candidate?._nativeNode ?? null;
      }} style={styles.canvas} />
      <Text style={styles.note}>
        Scan {performance.capture_fps.toFixed(1)} fps · local {performance.local_detection_ms ?? '—'} ms · API {performance.tracking_latency_ms ?? '—'} ms / {performance.tracking_age_ms ?? '—'} ms ({performance.tracking_status})
        {'\n'}Map {stats.mapHz.toFixed(1)} Hz · render {stats.renderFps} fps · depth anchors {stats.depthObjects}
        {'\n'}Objects kept {OBJECT_MEMORY_MS / 1000}s · obstacle dots within {OBSTACLE_RANGE_M}m kept {OBSTACLE_MEMORY_MS / 1000}s (max {MAX_OBSTACLE_DOTS}).
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

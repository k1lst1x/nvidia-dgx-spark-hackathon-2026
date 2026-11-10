import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import * as THREE from 'three';
import type { ObstacleFrame, VisionObject, VisionPerformance } from '../src/vision';

type MapDetection = Pick<VisionObject, 'label' | 'confidence' | 'x' | 'y' | 'width' | 'height' | 'distance_m' | 'track_id' | 'spatial'>;
type Props = { detections: MapDetection[]; obstacles?: ObstacleFrame | null; performance: VisionPerformance };
type Anchor = { label: string; x: number; y: number; z: number; firstSeenAt: number; lastSeenAt: number; seenTick: number };
type Dot = { x: number; y: number; z: number; seenAt: number; hits: number };
type Nearest = { label: string; distance: number; direction: string } | null;
type MapStats = {
  liveObjects: number;
  rememberedObjects: number;
  depthObjects: number;
  dots: number;
  mapHz: number;
  renderFps: number;
  nearest: Nearest;
};
type ViewMode = 'top' | 'orbit';
type PanelSize = 'mini' | 'small' | 'large';

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
// Only what was seen this recently counts for the "Nearest" readout.
const NEAREST_WINDOW_MS = 2500;

// The camera's horizontal field of view drawn on the floor. 60° matches the
// server's uncalibrated webcam stand-in (point_cloud.camera_intrinsics).
const FOV_DEG = 60;
const RING_DISTANCES_M = [1, 2, 3, OBSTACLE_RANGE_M];

const BACKGROUND = new THREE.Color(0x07110c);
const NEAR_COLOR = new THREE.Color(0xff6b4a);
const FAR_COLOR = new THREE.Color(0x56c8ff);
const LIVE_OBJECT_COLOR = new THREE.Color(0xb7ff56);
const REMEMBERED_OBJECT_COLOR = new THREE.Color(0x7d9686);

const VIEW_STORAGE_KEY = 'guidelens_live_map_view';
const SIZE_STORAGE_KEY = 'guidelens_live_map_size';

function readPreference<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const value = window.localStorage.getItem(key);
    return value && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
  } catch {
    return fallback;
  }
}

function writePreference(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Private windows may refuse storage; the map still works.
  }
}

function textSprite(text: string, options: { color?: string; background?: string; border?: string; height?: number } = {}) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return new THREE.Sprite();
  const font = '700 30px sans-serif';
  context.font = font;
  const width = Math.min(512, Math.ceil(context.measureText(text).width) + 28);
  canvas.width = width;
  canvas.height = 48;
  context.font = font;
  if (options.background) {
    context.fillStyle = options.background;
    context.beginPath();
    context.roundRect?.(2, 2, width - 4, 44, 10);
    if (!context.roundRect) context.rect(2, 2, width - 4, 44);
    context.fill();
    if (options.border) {
      context.strokeStyle = options.border;
      context.lineWidth = 2;
      context.stroke();
    }
  }
  context.fillStyle = options.color ?? '#F4FFE4';
  context.textBaseline = 'middle';
  context.fillText(text, 14, 25);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  const height = options.height ?? 0.26;
  sprite.scale.set((height * width) / 48, height, 1);
  sprite.userData.baseScale = [(height * width) / 48, height];
  sprite.renderOrder = 10;
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

/** Clock-free direction words for a scene point (forward is -z). */
function directionOf(point: { x: number; z: number }) {
  const angle = (Math.atan2(point.x, -point.z) * 180) / Math.PI;
  if (Math.abs(angle) <= 12) return 'ahead';
  if (Math.abs(angle) <= 40) return angle < 0 ? 'ahead-left' : 'ahead-right';
  return angle < 0 ? 'left' : 'right';
}

function urgencyColor(distance: number) {
  if (distance < 1.5) return '#FF8A6B';
  if (distance < 3) return '#FFD043';
  return '#B7FF56';
}

function ringLine(radius: number, color: number, opacity: number) {
  const points = Array.from({ length: 97 }, (_, index) => {
    const angle = (index / 96) * Math.PI * 2;
    return new THREE.Vector3(Math.cos(angle) * radius, 0.01, Math.sin(angle) * radius);
  });
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
}

export default function LiveSceneMap({ detections, obstacles = null, performance }: Props) {
  const window_ = useWindowDimensions();
  const host = useRef<HTMLElement | null>(null);
  const detectionsRef = useRef(detections);
  const obstaclesRef = useRef(obstacles);
  const resetRef = useRef<() => void>(() => {});
  const resetViewRef = useRef<() => void>(() => {});
  const [view, setView] = useState<ViewMode>(() => readPreference(VIEW_STORAGE_KEY, ['top', 'orbit'] as const, 'top'));
  const [size, setSize] = useState<PanelSize>(() => readPreference(SIZE_STORAGE_KEY, ['mini', 'small', 'large'] as const, 'small'));
  const [showDetails, setShowDetails] = useState(false);
  const viewRef = useRef(view);
  const [stats, setStats] = useState<MapStats>({
    liveObjects: 0, rememberedObjects: 0, depthObjects: 0, dots: 0, mapHz: 0, renderFps: 0, nearest: null,
  });
  detectionsRef.current = detections;
  obstaclesRef.current = obstacles;
  viewRef.current = view;

  useEffect(() => writePreference(VIEW_STORAGE_KEY, view), [view]);
  useEffect(() => writePreference(SIZE_STORAGE_KEY, size), [size]);

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = BACKGROUND.clone();
    scene.fog = new THREE.Fog(0x07110c, 9, 20);
    const camera = new THREE.PerspectiveCamera(48, 1, 0.05, 60);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.touchAction = 'none';
    container.appendChild(renderer.domElement);

    // --- Static floor: faint grid, distance rings with labels, field of view.
    const floor = new THREE.Group();
    scene.add(floor);
    const grid = new THREE.GridHelper(12, 24, 0x12271d, 0x0f2119);
    floor.add(grid);
    RING_DISTANCES_M.forEach((radius) => {
      const outer = radius === OBSTACLE_RANGE_M;
      floor.add(ringLine(radius, outer ? 0x4f8a64 : 0x2b4a39, outer ? 0.95 : 0.8));
      const label = textSprite(`${radius} m`, { color: outer ? '#9FD9A8' : '#6E9A7E', height: 0.22 });
      const angle = (24 * Math.PI) / 180;
      label.position.set(Math.sin(angle) * radius + 0.18, 0.05, -Math.cos(angle) * radius);
      floor.add(label);
    });
    const fovRadians = (FOV_DEG * Math.PI) / 180;
    const fov = new THREE.Mesh(
      new THREE.CircleGeometry(OBSTACLE_RANGE_M, 48, Math.PI / 2 - fovRadians / 2, fovRadians),
      new THREE.MeshBasicMaterial({ color: 0xb7ff56, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }),
    );
    fov.rotation.x = -Math.PI / 2;
    fov.position.y = 0.005;
    floor.add(fov);
    const fovEdges = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-Math.sin(fovRadians / 2) * OBSTACLE_RANGE_M, 0.012, -Math.cos(fovRadians / 2) * OBSTACLE_RANGE_M),
        new THREE.Vector3(0, 0.012, 0),
        new THREE.Vector3(Math.sin(fovRadians / 2) * OBSTACLE_RANGE_M, 0.012, -Math.cos(fovRadians / 2) * OBSTACLE_RANGE_M),
      ]),
      new THREE.LineBasicMaterial({ color: 0x6fae4f, transparent: true, opacity: 0.6 }),
    );
    floor.add(fovEdges);

    // The user: a heading chevron on the floor and a marker at eye height.
    const chevronShape = new THREE.Shape();
    chevronShape.moveTo(0, 0.34);
    chevronShape.lineTo(0.22, -0.2);
    chevronShape.lineTo(0, -0.08);
    chevronShape.lineTo(-0.22, -0.2);
    chevronShape.closePath();
    const chevron = new THREE.Mesh(new THREE.ShapeGeometry(chevronShape), new THREE.MeshBasicMaterial({ color: 0xb7ff56, side: THREE.DoubleSide }));
    chevron.rotation.x = -Math.PI / 2;
    chevron.position.y = 0.02;
    floor.add(chevron);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: 0xb7ff56 }));
    eye.position.set(0, EYE_HEIGHT_M, 0);
    floor.add(eye);
    const eyeStem = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, EYE_HEIGHT_M, 0)]),
      new THREE.LineBasicMaterial({ color: 0x4f8a64 }),
    );
    floor.add(eyeStem);

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
    const dotMaterial = new THREE.PointsMaterial({ size: 0.09, vertexColors: true, sizeAttenuation: true, depthWrite: false });
    const dotCloud = new THREE.Points(dotGeometry, dotMaterial);
    dotCloud.frustumCulled = false;
    scene.add(dotCloud);

    type ObjectVisual = { group: THREE.Group; marker: THREE.Mesh; stem: THREE.Line; base: THREE.Mesh; label: THREE.Sprite; text: string };
    const anchors = new Map<string, Anchor>();
    const anchorVisuals = new Map<string, ObjectVisual>();
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

    const createVisual = (): ObjectVisual => {
      const group = new THREE.Group();
      const marker = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.13),
        new THREE.MeshBasicMaterial({ color: LIVE_OBJECT_COLOR, transparent: true }),
      );
      const stem = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]),
        new THREE.LineBasicMaterial({ color: LIVE_OBJECT_COLOR, transparent: true }),
      );
      const base = new THREE.Mesh(
        new THREE.RingGeometry(0.16, 0.22, 32),
        new THREE.MeshBasicMaterial({ color: LIVE_OBJECT_COLOR, transparent: true, side: THREE.DoubleSide, depthWrite: false }),
      );
      base.rotation.x = -Math.PI / 2;
      const label = new THREE.Sprite();
      group.add(stem, base, marker, label);
      objectGroup.add(group);
      return { group, marker, stem, base, label, text: '' };
    };

    const syncObjects = (now: number) => {
      anchors.forEach((anchor, key) => {
        let visual = anchorVisuals.get(key);
        if (!visual) {
          visual = createVisual();
          anchorVisuals.set(key, visual);
        }
        const age = now - anchor.lastSeenAt;
        const live = age <= OBJECT_FRESH_MS;
        // Remembered objects dim from 70% to 15% over their memory window.
        const memoryProgress = live ? 0 : Math.min(1, (age - OBJECT_FRESH_MS) / (OBJECT_MEMORY_MS - OBJECT_FRESH_MS));
        const opacity = live ? 1 : 0.7 - memoryProgress * 0.55;
        const color = live ? LIVE_OBJECT_COLOR : REMEMBERED_OBJECT_COLOR;
        const height = Math.max(0.05, anchor.y);
        visual.group.position.set(anchor.x, 0, anchor.z);
        visual.marker.position.y = height;
        visual.stem.scale.y = height;
        visual.base.position.y = 0.015;
        [visual.marker, visual.stem, visual.base].forEach((part) => {
          const material = part.material as THREE.MeshBasicMaterial | THREE.LineBasicMaterial;
          material.color.copy(color);
          material.opacity = part === visual!.base ? opacity * 0.8 : opacity;
        });
        // Re-draw the label only when its text changes (0.1 m steps).
        const distance = distanceFromCamera(anchor);
        const text = `${anchor.label} · ${distance.toFixed(1)} m${live ? '' : ' · seen earlier'}`;
        if (text !== visual.text) {
          visual.group.remove(visual.label);
          disposeObject(visual.label);
          visual.label = textSprite(text, {
            color: live ? '#F4FFE4' : '#B7C9BC',
            background: 'rgba(8, 16, 12, 0.88)',
            border: live ? '#B7FF56' : '#4E6B58',
          });
          visual.group.add(visual.label);
          visual.text = text;
        }
        (visual.label.material as THREE.SpriteMaterial).opacity = live ? 1 : Math.max(0.35, opacity);
        visual.label.position.set(0, height + 0.32, 0);
      });
      anchorVisuals.forEach((visual, key) => {
        if (anchors.has(key)) return;
        objectGroup.remove(visual.group);
        disposeObject(visual.group);
        anchorVisuals.delete(key);
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
            const visual = anchorVisuals.get(matched);
            if (visual) {
              anchorVisuals.delete(matched);
              anchorVisuals.set(trackKey, visual);
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

    /** The closest thing seen just now, by distance along the floor. */
    const findNearest = (now: number): Nearest => {
      let best: Nearest = null;
      const consider = (label: string, point: { x: number; z: number }) => {
        const distance = Math.hypot(point.x, point.z);
        if (!best || distance < best.distance) best = { label, distance, direction: directionOf(point) };
      };
      anchors.forEach((anchor) => { if (now - anchor.lastSeenAt <= NEAREST_WINDOW_MS) consider(anchor.label, anchor); });
      dots.forEach((dot) => { if (now - dot.seenAt <= NEAREST_WINDOW_MS) consider('obstacle', dot); });
      return best;
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
      const nearest = findNearest(now);
      setStats((previous) => ({
        ...previous,
        liveObjects,
        rememberedObjects: anchors.size - liveObjects,
        depthObjects,
        dots: dotCount,
        mapHz: measuredMapHz,
        nearest,
      }));
    };

    // --- Camera: a top-down radar (forward is up) or a free orbit.
    const pivot = new THREE.Vector3(0, 0.4, -2.0);
    let orbitYaw = 0;
    let orbitPitch = 0.55;
    let orbitDistance = 6.2;
    const desiredPosition = new THREE.Vector3();
    const desiredTarget = new THREE.Vector3();
    const currentTarget = pivot.clone();
    const desiredUp = new THREE.Vector3();
    const updateDesiredCamera = () => {
      if (viewRef.current === 'top') {
        desiredPosition.set(0, 8.0, -1.8);
        desiredTarget.set(0, 0, -1.8);
        desiredUp.set(0, 0, -1);
      } else {
        desiredPosition.set(
          pivot.x + Math.sin(orbitYaw) * Math.cos(orbitPitch) * orbitDistance,
          pivot.y + Math.sin(orbitPitch) * orbitDistance,
          pivot.z + Math.cos(orbitYaw) * Math.cos(orbitPitch) * orbitDistance,
        );
        desiredTarget.copy(pivot);
        desiredUp.set(0, 1, 0);
      }
    };
    updateDesiredCamera();
    camera.position.copy(desiredPosition);
    camera.up.copy(desiredUp);
    currentTarget.copy(desiredTarget);
    camera.lookAt(currentTarget);

    let dragging: { x: number; y: number } | null = null;
    const onPointerDown = (event: PointerEvent) => {
      if (viewRef.current !== 'orbit') return;
      dragging = { x: event.clientX, y: event.clientY };
      renderer.domElement.setPointerCapture?.(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      orbitYaw -= (event.clientX - dragging.x) * 0.01;
      orbitPitch = Math.min(1.45, Math.max(0.12, orbitPitch + (event.clientY - dragging.y) * 0.008));
      dragging = { x: event.clientX, y: event.clientY };
    };
    const onPointerUp = () => { dragging = null; };
    const onWheel = (event: WheelEvent) => {
      if (viewRef.current !== 'orbit') return;
      event.preventDefault();
      orbitDistance = Math.min(12, Math.max(3, orbitDistance * (event.deltaY > 0 ? 1.08 : 0.93)));
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointercancel', onPointerUp);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });
    resetViewRef.current = () => {
      orbitYaw = 0;
      orbitPitch = 0.55;
      orbitDistance = 6.2;
    };

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width < 10 || height < 10) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      renderer.domElement.style.width = `${width}px`;
      renderer.domElement.style.height = `${height}px`;
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    const timer = window.setInterval(integrate, MAP_INTERVAL_MS);
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      // Hidden (mini) panel: keep the map's memory, skip drawing.
      if (container.clientHeight < 10) return;
      updateDesiredCamera();
      camera.position.lerp(desiredPosition, 0.18);
      camera.up.lerp(desiredUp, 0.18).normalize();
      currentTarget.lerp(desiredTarget, 0.18);
      camera.lookAt(currentTarget);
      // Keep object labels readable from straight above.
      const labelScale = viewRef.current === 'top' ? 1.7 : 1;
      anchorVisuals.forEach(({ label }) => {
        const base = label.userData.baseScale as [number, number] | undefined;
        if (base) label.scale.set(base[0] * labelScale, base[1] * labelScale, 1);
      });
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
      anchorVisuals.forEach((visual) => {
        objectGroup.remove(visual.group);
        disposeObject(visual.group);
      });
      anchorVisuals.clear();
      dots.clear();
      // Do not re-ingest the frame that was on screen when Clear was pressed.
      lastObstacleSequence = obstaclesRef.current?.sequence ?? lastObstacleSequence;
      dotGeometry.setDrawRange(0, 0);
      setStats((previous) => ({ ...previous, liveObjects: 0, rememberedObjects: 0, depthObjects: 0, dots: 0, nearest: null }));
    };

    return () => {
      window.clearInterval(timer);
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('pointercancel', onPointerUp);
      renderer.domElement.removeEventListener('wheel', onWheel);
      disposeObject(objectGroup);
      disposeObject(floor);
      dotGeometry.dispose();
      dotMaterial.dispose();
      renderer.dispose();
      container.replaceChildren();
    };
  }, []);

  const panelWidth = size === 'large' ? Math.min(640, window_.width - 24) : Math.min(320, window_.width - 24);
  // Large fills the space between the top bar and the bottom card instead of
  // a fixed height, so it never slides under the camera screen's controls.
  const canvasStyle = size === 'mini' ? styles.canvasHidden : size === 'large' ? styles.canvasFill : styles.canvasSmall;
  const nearest = stats.nearest;
  const nearestText = nearest
    ? `${nearest.label === 'obstacle' ? 'Obstacle' : nearest.label} ${nearest.distance.toFixed(1)} m ${nearest.direction}`
    : `Nothing seen within ${OBSTACLE_RANGE_M} m`;

  return (
    <View style={[styles.panel, { width: panelWidth }, size === 'large' && styles.panelFill]}>
      <View style={styles.header}>
        <View style={styles.titleBlock}>
          <Text style={styles.kicker}>LIVE MAP</Text>
          <Text style={styles.subtle}>camera-relative</Text>
        </View>
        <View style={styles.segment} accessibilityRole="radiogroup" accessibilityLabel="Map view">
          {(['top', 'orbit'] as const).map((mode) => (
            <Pressable
              key={mode}
              onPress={() => setView(mode)}
              style={[styles.segmentButton, view === mode && styles.segmentButtonActive]}
              accessibilityRole="radio"
              accessibilityState={{ selected: view === mode }}
              accessibilityLabel={mode === 'top' ? 'Top-down view' : '3D view, drag to rotate'}
            >
              <Text style={[styles.segmentText, view === mode && styles.segmentTextActive]}>{mode === 'top' ? 'Top' : '3D'}</Text>
            </Pressable>
          ))}
        </View>
        <Pressable
          onPress={() => setSize(size === 'large' ? 'small' : 'large')}
          style={styles.iconButton}
          accessibilityRole="button"
          accessibilityLabel={size === 'large' ? 'Make the map smaller' : 'Make the map larger'}
        >
          <Text style={styles.iconButtonText}>{size === 'large' ? '⤡' : '⤢'}</Text>
        </Pressable>
        <Pressable
          onPress={() => setSize(size === 'mini' ? 'small' : 'mini')}
          style={styles.iconButton}
          accessibilityRole="button"
          accessibilityLabel={size === 'mini' ? 'Show the map' : 'Hide the map'}
        >
          <Text style={styles.iconButtonText}>{size === 'mini' ? '▾' : '▴'}</Text>
        </Pressable>
      </View>

      <View style={styles.nearestRow} accessibilityLiveRegion="polite" accessibilityLabel={`Nearest: ${nearestText}`}>
        <View style={[styles.nearestDot, { backgroundColor: nearest ? urgencyColor(nearest.distance) : '#4E6B58' }]} />
        <Text style={styles.nearestLabel}>NEAREST</Text>
        <Text style={[styles.nearestText, { color: nearest ? urgencyColor(nearest.distance) : '#B7C9BC' }]} numberOfLines={1}>
          {nearestText}
        </Text>
      </View>

      <View
        ref={(node) => {
          const candidate = node as unknown as HTMLElement | { _nativeNode?: HTMLElement } | null;
          host.current = candidate && 'appendChild' in candidate ? candidate : candidate?._nativeNode ?? null;
        }}
        style={[styles.canvas, canvasStyle]}
      />

      {size !== 'mini' && (
        <>
          <View style={styles.legend}>
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: '#FF6B4A' }]} /><Text style={styles.legendText}>near</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: '#56C8FF' }]} /><Text style={styles.legendText}>far surface</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, styles.swatchDiamond, { backgroundColor: '#B7FF56' }]} /><Text style={styles.legendText}>object</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, styles.swatchDiamond, { backgroundColor: '#7D9686' }]} /><Text style={styles.legendText}>seen earlier</Text></View>
            {view === 'orbit' && (
              <Pressable onPress={() => resetViewRef.current()} accessibilityRole="button" accessibilityLabel="Reset the 3D view">
                <Text style={styles.linkText}>reset view</Text>
              </Pressable>
            )}
          </View>

          <View style={styles.footer}>
            <Text style={styles.counts}>
              {stats.liveObjects} live · {stats.rememberedObjects} earlier · {stats.dots} dots
            </Text>
            <Pressable onPress={() => setShowDetails((value) => !value)} accessibilityRole="button" accessibilityLabel="Show map details">
              <Text style={styles.linkText}>{showDetails ? 'hide details' : 'details'}</Text>
            </Pressable>
            <Pressable onPress={() => resetRef.current()} style={styles.clear} accessibilityRole="button" accessibilityLabel="Clear the map">
              <Text style={styles.clearText}>Clear</Text>
            </Pressable>
          </View>

          {showDetails && (
            <Text style={styles.note}>
              Scan {performance.capture_fps.toFixed(1)} fps · local {performance.local_detection_ms ?? '—'} ms · API {performance.tracking_latency_ms ?? '—'} ms / {performance.tracking_age_ms ?? '—'} ms ({performance.tracking_status})
              {'\n'}Map {stats.mapHz.toFixed(1)} Hz · render {stats.renderFps} fps · objects with depth {stats.depthObjects}
              {'\n'}Objects kept {OBJECT_MEMORY_MS / 1000}s · surfaces within {OBSTACLE_RANGE_M} m kept {OBSTACLE_MEMORY_MS / 1000}s (max {MAX_OBSTACLE_DOTS}).
              {'\n'}Moves with the camera; not a saved room map. Objects without depth are not shown.
            </Text>
          )}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    position: 'absolute', right: 0, top: 0, backgroundColor: 'rgba(5, 12, 9, 0.92)',
    borderColor: 'rgba(183, 255, 86, 0.4)', borderWidth: 1, borderRadius: 16, padding: 10, gap: 8, zIndex: 6,
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  titleBlock: { flex: 1 },
  kicker: { color: '#B7FF56', fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  subtle: { color: '#7D9686', fontSize: 10, fontWeight: '600' },
  segment: { flexDirection: 'row', backgroundColor: '#0F2119', borderRadius: 9, padding: 2 },
  segmentButton: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 7 },
  segmentButtonActive: { backgroundColor: '#B7FF56' },
  segmentText: { color: '#B7C9BC', fontSize: 11, fontWeight: '800' },
  segmentTextActive: { color: '#07110C' },
  iconButton: { width: 28, height: 28, borderRadius: 8, backgroundColor: '#0F2119', alignItems: 'center', justifyContent: 'center' },
  iconButtonText: { color: '#D5E6D8', fontSize: 14, fontWeight: '800' },
  nearestRow: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: '#0B1A13', borderRadius: 10, paddingHorizontal: 10, paddingVertical: 7 },
  nearestDot: { width: 8, height: 8, borderRadius: 4 },
  nearestLabel: { color: '#7D9686', fontSize: 10, fontWeight: '900', letterSpacing: 0.8 },
  nearestText: { fontSize: 14, fontWeight: '800', flex: 1 },
  canvas: { borderRadius: 12, overflow: 'hidden', backgroundColor: '#07110C' },
  canvasSmall: { height: 200 },
  canvasFill: { flex: 1, minHeight: 160 },
  canvasHidden: { height: 0, opacity: 0 },
  panelFill: { bottom: 0 },
  legend: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: 10, rowGap: 4 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  swatch: { width: 8, height: 8, borderRadius: 4 },
  swatchDiamond: { borderRadius: 1, transform: [{ rotate: '45deg' }] },
  legendText: { color: '#B7C9BC', fontSize: 10, fontWeight: '600' },
  footer: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  counts: { color: '#D5E6D8', fontSize: 11, fontWeight: '700', flex: 1 },
  linkText: { color: '#8FD3FF', fontSize: 10, fontWeight: '700' },
  clear: { backgroundColor: '#163024', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  clearText: { color: '#E7F6EA', fontSize: 11, fontWeight: '800' },
  note: { color: '#B7C9BC', fontSize: 10, lineHeight: 14 },
});

import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import * as THREE from 'three';
import type { BlockedFrame, ObstacleFrame, VisionObject, VisionPerformance } from '../src/vision';

type MapDetection = Pick<VisionObject, 'label' | 'confidence' | 'x' | 'y' | 'width' | 'height' | 'distance_m' | 'track_id' | 'spatial'>;
type Props = { detections: MapDetection[]; obstacles?: ObstacleFrame | null; blocked?: BlockedFrame | null; performance: VisionPerformance };
type Anchor = {
  label: string;
  x: number; y: number; z: number;
  /** Smoothed position actually drawn, so pins glide instead of jumping at 4 Hz. */
  sx: number; sy: number; sz: number;
  firstSeenAt: number;
  /** Last time a measured distance confirmed this object. */
  lastSeenAt: number;
  /** Last time the detector saw it in this direction without a distance. */
  bearingSeenAt: number;
  seenTick: number;
};
/** A detection with a direction but no measured distance. */
type Ray = { label: string; angle: number; shown: number; seenAt: number };
type Dot = { x: number; y: number; z: number; seenAt: number; hits: number };
type Sighting = { label: string; distance: number; bearing: number; direction: string };
type MapStats = {
  liveObjects: number;
  rememberedObjects: number;
  directionOnly: number;
  depthObjects: number;
  dots: number;
  mapHz: number;
  renderFps: number;
  nearest: Sighting | null;
  others: Sighting[];
  inView: string[];
  blockedView: { direction: string; bearing: number; covered: boolean; dark: boolean } | null;
};
type ViewMode = 'top' | 'orbit';
type PanelSize = 'mini' | 'small' | 'large';

const EYE_HEIGHT_M = 1.45;
const MAP_INTERVAL_MS = 100;

// Object memory: a detected object stays on the map after it leaves view,
// fading out, then is forgotten. The cap keeps the scene small.
const OBJECT_FRESH_MS = 2000;
const OBJECT_MEMORY_MS = 15_000;
const MAX_OBJECT_ANCHORS = 24;
// Untracked detections (or a tracker that re-issued an ID) are merged with a
// remembered anchor of the same label within this radius.
const OBJECT_MATCH_RADIUS_M = 0.8;
// A direction-only detection within this angle of a remembered object of the
// same label is treated as that object still being in view.
const BEARING_MATCH_DEG = 12;
const MAX_RAYS = 8;

// Obstacle dots: unlabeled surfaces (walls, trees, poles) from metric depth.
// Only near surfaces are kept, each is forgotten after a few seconds, dots are
// merged on a voxel grid, and the total is hard-capped.
const OBSTACLE_RANGE_M = 4.5;
const OBSTACLE_MEMORY_MS = 8000;
const OBSTACLE_VOXEL_M = 0.15;
const MAX_OBSTACLE_DOTS = 1200;
// Blocked view: directions where the camera frame is featureless or dark and
// no object was named (src/blockedView.ts), such as a plain wall, a trunk or a
// covered lens. The distance is unknown, so these dots line the direction
// instead of sitting at a measured spot, and a measured surface replaces them.
const BLOCKED_MEMORY_MS = 4000;
const BLOCKED_FRESH_MS = 1500;
const BLOCKED_RADII_M = [0.6, 0.9, 1.2, 1.5, 1.8];
const BLOCKED_HEIGHTS_M = [0.45, 0.95, 1.45];
const MAX_BLOCKED_COLUMNS = 16;
const BLOCKED_COLOR = new THREE.Color(0xe0b45c);
// Only what was measured this recently counts for the "Nearest" readout.
const NEAREST_WINDOW_MS = 2500;
const DANGER_M = 1.5;
const CAUTION_M = 3;

// The camera's horizontal field of view drawn on the floor. 60° matches the
// server's uncalibrated webcam stand-in (point_cloud.camera_intrinsics).
const FOV_DEG = 60;
const FOV_RAD = (FOV_DEG * Math.PI) / 180;
const RING_DISTANCES_M = [1, 2, 3, OBSTACLE_RANGE_M];
const RING_LABEL_BEARING = (-52 * Math.PI) / 180;

const BACKGROUND = new THREE.Color(0x060f0b);
const FLOOR_CENTER = new THREE.Color(0x10281d);
const NEAR_COLOR = new THREE.Color(0xff6b4a);
const FAR_COLOR = new THREE.Color(0x56c8ff);
const BLACK = new THREE.Color(0x000000);
const URGENT = { near: '#FF6B4A', mid: '#FFD043', far: '#B7FF56', memory: '#7D9686', ray: '#8FD3FF' };

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

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    (Array.isArray(material) ? material : material ? [material] : []).forEach((item) => {
      (item as THREE.PointsMaterial).map?.dispose?.();
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

/** Radians, positive to the right of straight ahead (forward is -z). */
function bearingOf(point: { x: number; z: number }) {
  return Math.atan2(point.x, -point.z);
}

/** Clock-free direction words for a bearing. */
function directionWords(bearing: number) {
  const angle = (bearing * 180) / Math.PI;
  if (Math.abs(angle) <= 12) return 'ahead';
  if (Math.abs(angle) <= 40) return angle < 0 ? 'ahead-left' : 'ahead-right';
  if (Math.abs(angle) <= 135) return angle < 0 ? 'left' : 'right';
  return 'behind';
}

function urgencyColor(distance: number) {
  if (distance < DANGER_M) return URGENT.near;
  if (distance < CAUTION_M) return URGENT.mid;
  return URGENT.far;
}

function onFloor(radius: number, bearing: number, y = 0.01) {
  return new THREE.Vector3(Math.sin(bearing) * radius, y, -Math.cos(bearing) * radius);
}

function ringLine(radius: number, color: number, opacity: number) {
  const points = Array.from({ length: 129 }, (_, index) => onFloor(radius, (index / 128) * Math.PI * 2));
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
}

/** A floor sector (forward-centered) whose color fades from `center` at the origin to `rim`. */
function fadedSector(radius: number, width: number, center: THREE.Color, rim: THREE.Color, segments = 48) {
  const geometry = new THREE.CircleGeometry(radius, segments, Math.PI / 2 - width / 2, width);
  const positions = geometry.getAttribute('position');
  const colors = new Float32Array(positions.count * 3);
  const mixed = new THREE.Color();
  for (let index = 0; index < positions.count; index += 1) {
    const r = Math.min(1, Math.hypot(positions.getX(index), positions.getY(index)) / radius);
    mixed.copy(rim).lerp(center, Math.pow(1 - r, 0.85));
    colors.set([mixed.r, mixed.g, mixed.b], index * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.rotateX(-Math.PI / 2);
  return geometry;
}

function glowMaterial(opacity = 1) {
  return new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
}

function dotTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.45, 'rgba(255,255,255,0.85)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 64);
  }
  return new THREE.CanvasTexture(canvas);
}

type LabelKind = 'object' | 'ray' | 'ring' | 'compass';
type MapLabel = {
  el: HTMLDivElement;
  kind: LabelKind;
  signature: string;
  width: number;
  height: number;
  priority: number;
  world: () => THREE.Vector3;
};

const LABEL_BASE_STYLE = [
  'position:absolute', 'left:0', 'top:0', 'white-space:nowrap', 'pointer-events:none', 'will-change:transform',
  'font-family:system-ui,-apple-system,Segoe UI,sans-serif', 'transition:opacity 120ms linear',
].join(';');

export default function LiveSceneMap({ detections, obstacles = null, blocked = null, performance }: Props) {
  const window_ = useWindowDimensions();
  const host = useRef<HTMLElement | null>(null);
  const detectionsRef = useRef(detections);
  const obstaclesRef = useRef(obstacles);
  const blockedRef = useRef(blocked);
  const resetRef = useRef<() => void>(() => {});
  const resetViewRef = useRef<() => void>(() => {});
  const [view, setView] = useState<ViewMode>(() => readPreference(VIEW_STORAGE_KEY, ['top', 'orbit'] as const, 'top'));
  const [size, setSize] = useState<PanelSize>(() => readPreference(SIZE_STORAGE_KEY, ['mini', 'small', 'large'] as const, 'small'));
  const [showDetails, setShowDetails] = useState(false);
  const viewRef = useRef(view);
  const [stats, setStats] = useState<MapStats>({
    liveObjects: 0, rememberedObjects: 0, directionOnly: 0, depthObjects: 0, dots: 0, mapHz: 0, renderFps: 0,
    nearest: null, others: [], inView: [], blockedView: null,
  });
  detectionsRef.current = detections;
  obstaclesRef.current = obstacles;
  blockedRef.current = blocked;
  viewRef.current = view;

  useEffect(() => writePreference(VIEW_STORAGE_KEY, view), [view]);
  useEffect(() => writePreference(SIZE_STORAGE_KEY, size), [size]);

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;
    const reducedMotion = Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);

    const scene = new THREE.Scene();
    scene.background = BACKGROUND.clone();
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pixelRatio);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.touchAction = 'none';
    container.appendChild(renderer.domElement);
    // Crisp HTML labels drawn over the canvas; placed every frame without overlaps.
    const labelLayer = document.createElement('div');
    labelLayer.style.cssText = 'position:absolute;inset:0;overflow:hidden;pointer-events:none';
    container.appendChild(labelLayer);

    // Top view: a true radar (orthographic, forward is up, no parallax).
    const topCamera = new THREE.OrthographicCamera(-5, 5, 3, -3, 0.1, 60);
    const TOP_CENTER_Z = -(OBSTACLE_RANGE_M - 0.9) / 2 - 0.2;
    let topZoom = 1;
    // 3D view: an orbit around the space ahead.
    const orbitCamera = new THREE.PerspectiveCamera(46, 1, 0.05, 80);
    let viewWidth = 1;
    let viewHeight = 1;

    // --- Floor: soft radial glow, polar grid, distance rings, field of view.
    const floor = new THREE.Group();
    scene.add(floor);
    const floorDisc = new THREE.Mesh(
      fadedSector(OBSTACLE_RANGE_M + 1.2, Math.PI * 2, FLOOR_CENTER, BACKGROUND, 96),
      new THREE.MeshBasicMaterial({ vertexColors: true, depthWrite: false }),
    );
    floorDisc.position.y = -0.002;
    floor.add(floorDisc);
    const spokes: THREE.Vector3[] = [];
    for (let degrees = 0; degrees < 360; degrees += 30) {
      const bearing = (degrees * Math.PI) / 180;
      spokes.push(onFloor(0.45, bearing, 0.004), onFloor(OBSTACLE_RANGE_M, bearing, 0.004));
    }
    floor.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(spokes),
      new THREE.LineBasicMaterial({ color: 0x1a3a2b, transparent: true, opacity: 0.55 }),
    ));
    RING_DISTANCES_M.forEach((radius) => {
      const outer = radius === OBSTACLE_RANGE_M;
      floor.add(ringLine(radius, outer ? 0x4f8a64 : 0x24453a, outer ? 1 : 0.85));
    });
    // Everything inside DANGER_M glows red while something measured is that close.
    const dangerMaterial = glowMaterial(0);
    const dangerZone = new THREE.Mesh(fadedSector(DANGER_M, Math.PI * 2, new THREE.Color(0x5a1408), new THREE.Color(0x2a0804), 64), dangerMaterial);
    dangerZone.position.y = 0.003;
    floor.add(dangerZone);
    const fov = new THREE.Mesh(fadedSector(OBSTACLE_RANGE_M, FOV_RAD, new THREE.Color(0x2c4a12), BLACK), glowMaterial());
    fov.position.y = 0.005;
    floor.add(fov);
    floor.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        onFloor(OBSTACLE_RANGE_M, -FOV_RAD / 2, 0.012), new THREE.Vector3(0, 0.012, 0), onFloor(OBSTACLE_RANGE_M, FOV_RAD / 2, 0.012),
      ]),
      new THREE.LineBasicMaterial({ color: 0x6fae4f, transparent: true, opacity: 0.7 }),
    ));
    // A slow sweep across the field of view: shows the map is live, not frozen.
    const sweep = new THREE.Mesh(fadedSector(OBSTACLE_RANGE_M, (7 * Math.PI) / 180, new THREE.Color(0x3d6b1a), BLACK, 12), glowMaterial());
    sweep.position.y = 0.007;
    sweep.visible = !reducedMotion;
    floor.add(sweep);

    // The user: a heading chevron with a halo, and an eye marker for the 3D view.
    const chevronShape = new THREE.Shape();
    chevronShape.moveTo(0, 0.3);
    chevronShape.lineTo(0.2, -0.18);
    chevronShape.lineTo(0, -0.07);
    chevronShape.lineTo(-0.2, -0.18);
    chevronShape.closePath();
    const chevronGeometry = new THREE.ShapeGeometry(chevronShape);
    chevronGeometry.rotateX(-Math.PI / 2);
    const chevron = new THREE.Mesh(chevronGeometry, new THREE.MeshBasicMaterial({ color: 0xb7ff56, side: THREE.DoubleSide }));
    chevron.position.y = 0.02;
    floor.add(chevron);
    const userHaloGeometry = new THREE.RingGeometry(0.34, 0.38, 48);
    userHaloGeometry.rotateX(-Math.PI / 2);
    const userHalo = new THREE.Mesh(userHaloGeometry, new THREE.MeshBasicMaterial({ color: 0xb7ff56, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
    userHalo.position.y = 0.015;
    floor.add(userHalo);
    const eye = new THREE.Group();
    eye.add(
      new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: 0xb7ff56 })),
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -EYE_HEIGHT_M, 0), new THREE.Vector3(0, 0, 0)]),
        new THREE.LineBasicMaterial({ color: 0x4f8a64 }),
      ),
    );
    eye.position.set(0, EYE_HEIGHT_M, 0);
    floor.add(eye);

    const objectGroup = new THREE.Group();
    scene.add(objectGroup);
    const rayGroup = new THREE.Group();
    scene.add(rayGroup);

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
    const dotMaterial = new THREE.PointsMaterial({
      size: 0.12, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const dotCloud = new THREE.Points(dotGeometry, dotMaterial);
    dotCloud.frustumCulled = false;
    scene.add(dotCloud);

    // Blocked-view dots: a fixed, small buffer (columns x radii x heights).
    const blockedCapacity = MAX_BLOCKED_COLUMNS * BLOCKED_RADII_M.length * BLOCKED_HEIGHTS_M.length;
    const blockedPositions = new Float32Array(blockedCapacity * 3);
    const blockedColors = new Float32Array(blockedCapacity * 3);
    const blockedGeometry = new THREE.BufferGeometry();
    const blockedPositionAttribute = new THREE.BufferAttribute(blockedPositions, 3).setUsage(THREE.DynamicDrawUsage);
    const blockedColorAttribute = new THREE.BufferAttribute(blockedColors, 3).setUsage(THREE.DynamicDrawUsage);
    blockedGeometry.setAttribute('position', blockedPositionAttribute);
    blockedGeometry.setAttribute('color', blockedColorAttribute);
    blockedGeometry.setDrawRange(0, 0);
    const blockedMaterial = new THREE.PointsMaterial({
      size: 0.085, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false,
    });
    const blockedCloud = new THREE.Points(blockedGeometry, blockedMaterial);
    blockedCloud.frustumCulled = false;
    scene.add(blockedCloud);
    const blockedDirs = new Map<number, { bearing: number; seenAt: number }>();
    let blockedState = { covered: false, dark: false };
    let lastBlockedAt = -1;
    const blockedLabelPoint = new THREE.Vector3();
    const blockedColor = new THREE.Color();

    type ObjectVisual = { group: THREE.Group; pin: THREE.Mesh; halo: THREE.Mesh; stem: THREE.Line; marker: THREE.Mesh; color: THREE.Color };
    type RayVisual = { line: THREE.Line; tip: THREE.Mesh };
    const anchors = new Map<string, Anchor>();
    const anchorVisuals = new Map<string, ObjectVisual>();
    const rays = new Map<string, Ray>();
    const rayVisuals = new Map<string, RayVisual>();
    // Insertion order doubles as age order: a refreshed dot is re-inserted at the end.
    const dots = new Map<string, Dot>();
    const labels = new Map<string, MapLabel>();
    let nearestKey: string | null = null;
    let lastObstacleSequence = -1;
    let anchorCounter = 0;
    let tickCounter = 0;
    let integrationCount = 0;
    let integrationStarted = performanceNow();
    let measuredMapHz = 0;
    let renderedFrames = 0;
    let renderStarted = performanceNow();
    let dangerLevel = 0;
    const fadeColor = new THREE.Color();
    const scratch = new THREE.Vector3();

    // ------------------------------------------------------------------ labels
    const ensureLabel = (key: string, kind: LabelKind, world: () => THREE.Vector3) => {
      let label = labels.get(key);
      if (!label) {
        const el = document.createElement('div');
        el.style.cssText = LABEL_BASE_STYLE;
        el.style.opacity = '0';
        labelLayer.appendChild(el);
        label = { el, kind, signature: '', width: 0, height: 0, priority: 0, world };
        labels.set(key, label);
      }
      label.world = world;
      return label;
    };

    /** Re-render a label only when its content changes, then measure it once. */
    const setLabel = (label: MapLabel, parts: Array<{ text: string; color: string; weight?: number }>, style: string) => {
      const signature = `${style}|${parts.map((part) => `${part.text}:${part.color}`).join('|')}`;
      if (signature === label.signature) return;
      label.signature = signature;
      label.el.style.cssText = `${LABEL_BASE_STYLE};${style};opacity:${label.el.style.opacity || 0}`;
      label.el.replaceChildren(...parts.map((part, index) => {
        const span = document.createElement('span');
        span.textContent = index ? ` ${part.text}` : part.text;
        span.style.color = part.color;
        span.style.fontWeight = String(part.weight ?? 600);
        return span;
      }));
      label.width = label.el.offsetWidth;
      label.height = label.el.offsetHeight;
    };

    const removeLabel = (key: string) => {
      const label = labels.get(key);
      if (!label) return;
      label.el.remove();
      labels.delete(key);
    };

    RING_DISTANCES_M.forEach((radius) => {
      const point = onFloor(radius, RING_LABEL_BEARING, 0.02);
      const label = ensureLabel(`ring:${radius}`, 'ring', () => point);
      label.priority = 500 + radius;
      setLabel(label, [{ text: `${radius} m`, color: radius === OBSTACLE_RANGE_M ? '#9FD9A8' : '#5F8A72', weight: 700 }], 'font-size:10px');
    });
    const aheadPoint = onFloor(OBSTACLE_RANGE_M + 0.32, 0, 0.02);
    const aheadLabel = ensureLabel('compass:ahead', 'compass', () => aheadPoint);
    aheadLabel.priority = 400;
    setLabel(aheadLabel, [{ text: 'AHEAD', color: '#7FB98F', weight: 800 }], 'font-size:9px;letter-spacing:1.5px');

    const placeLabels = (camera: THREE.Camera) => {
      const width = viewWidth;
      const height = viewHeight;
      const taken: Array<[number, number, number, number]> = [];
      const fits = (left: number, top: number, w: number, h: number) => {
        if (left < 2 || top < 2 || left + w > width - 2 || top + h > height - 2) return false;
        return taken.every(([l, t, r, b]) => left + w + 3 <= l || left >= r + 3 || top + h + 2 <= t || top >= b + 2);
      };
      const project = (point: THREE.Vector3) => {
        scratch.copy(point).project(camera);
        if (scratch.z < -1 || scratch.z > 1) return null;
        return [((scratch.x + 1) / 2) * width, ((1 - scratch.y) / 2) * height] as const;
      };
      // Keep the user's own marker clear of labels.
      const user = project(new THREE.Vector3(0, 0, 0));
      if (user) taken.push([user[0] - 16, user[1] - 16, user[0] + 16, user[1] + 16]);
      [...labels.values()].sort((left, right) => left.priority - right.priority).forEach((label) => {
        const anchorPoint = project(label.world());
        let placed: [number, number] | null = null;
        if (anchorPoint) {
          const [px, py] = anchorPoint;
          const w = label.width;
          const h = label.height;
          const candidates: Array<[number, number]> = label.kind === 'object' || label.kind === 'ray'
            ? [[px - w / 2, py - h - 10], [px + 11, py - h / 2], [px - w - 11, py - h / 2], [px - w / 2, py + 10]]
            : [[px - w / 2, py - h / 2]];
          placed = candidates.find(([left, top]) => fits(left, top, w, h)) ?? null;
          if (placed) taken.push([placed[0], placed[1], placed[0] + w, placed[1] + h]);
        }
        if (placed) {
          label.el.style.transform = `translate3d(${Math.round(placed[0])}px, ${Math.round(placed[1])}px, 0)`;
          label.el.style.opacity = '1';
        } else {
          label.el.style.opacity = '0';
        }
      });
    };

    // ----------------------------------------------------------------- objects
    const createVisual = (): ObjectVisual => {
      const group = new THREE.Group();
      const pinGeometry = new THREE.CircleGeometry(0.11, 32);
      pinGeometry.rotateX(-Math.PI / 2);
      const pin = new THREE.Mesh(pinGeometry, new THREE.MeshBasicMaterial({ color: 0xb7ff56, transparent: true, depthWrite: false }));
      pin.position.y = 0.03;
      const haloGeometry = new THREE.RingGeometry(0.17, 0.215, 40);
      haloGeometry.rotateX(-Math.PI / 2);
      const halo = new THREE.Mesh(haloGeometry, new THREE.MeshBasicMaterial({ color: 0xb7ff56, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      halo.position.y = 0.025;
      const stem = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]),
        new THREE.LineBasicMaterial({ color: 0xb7ff56, transparent: true }),
      );
      const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.12), new THREE.MeshBasicMaterial({ color: 0xb7ff56, transparent: true }));
      group.add(halo, pin, stem, marker);
      objectGroup.add(group);
      return { group, pin, halo, stem, marker, color: new THREE.Color(0xb7ff56) };
    };

    const objectLabelPoint = (anchor: Anchor) => () => (viewRef.current === 'top'
      ? scratchLabel.set(anchor.sx, 0, anchor.sz)
      : scratchLabel.set(anchor.sx, Math.max(0.05, anchor.sy) + 0.16, anchor.sz));
    const scratchLabel = new THREE.Vector3();

    const syncObjects = (now: number) => {
      anchors.forEach((anchor, key) => {
        let visual = anchorVisuals.get(key);
        if (!visual) {
          visual = createVisual();
          anchorVisuals.set(key, visual);
        }
        const measuredAge = now - anchor.lastSeenAt;
        const measured = measuredAge <= OBJECT_FRESH_MS;
        // Seen in this direction just now, but its distance is from earlier.
        const inView = !measured && now - anchor.bearingSeenAt <= OBJECT_FRESH_MS;
        const distance = distanceFromCamera(anchor);
        const memoryProgress = measured || inView ? 0 : Math.min(1, (measuredAge - OBJECT_FRESH_MS) / (OBJECT_MEMORY_MS - OBJECT_FRESH_MS));
        const opacity = measured ? 1 : inView ? 0.85 : 0.7 - memoryProgress * 0.5;
        visual.color.set(measured ? urgencyColor(distance) : inView ? '#D9F99D' : URGENT.memory);
        [visual.pin, visual.halo, visual.stem, visual.marker].forEach((part) => {
          const material = part.material as THREE.MeshBasicMaterial | THREE.LineBasicMaterial;
          material.color.copy(visual!.color);
          material.opacity = part === visual!.halo ? opacity * 0.6 : opacity;
        });

        const label = ensureLabel(`object:${key}`, 'object', objectLabelPoint(anchor));
        // Nearest measured objects claim label space first; memories last.
        label.priority = measured ? distance : inView ? 50 + distance : 100 + measuredAge / 1000;
        const name = anchor.label.charAt(0).toUpperCase() + anchor.label.slice(1);
        if (measured) {
          const color = urgencyColor(distance);
          const accent = key === nearestKey;
          setLabel(label, [{ text: name, color: '#EEF8F1' }, { text: `${distance.toFixed(1)} m`, color, weight: 800 }],
            `font-size:${accent ? 12 : 11}px;padding:3px 7px;border-radius:8px;background:rgba(6,14,10,.9);border:1px solid ${accent ? color : 'rgba(183,255,86,.35)'}`);
        } else if (inView) {
          setLabel(label, [{ text: name, color: '#E4F2E8' }, { text: `~${distance.toFixed(1)} m`, color: '#D9F99D', weight: 700 }],
            'font-size:11px;padding:3px 7px;border-radius:8px;background:rgba(6,14,10,.86);border:1px dashed rgba(217,249,157,.5)');
        } else {
          setLabel(label, [{ text: name, color: '#A9BDB0' }, { text: `· ${Math.max(1, Math.round(measuredAge / 1000))}s ago`, color: '#6F887A' }],
            'font-size:10px;padding:2px 6px;border-radius:7px;background:rgba(6,14,10,.72);border:1px solid rgba(125,150,134,.3)');
        }
      });
      anchorVisuals.forEach((visual, key) => {
        if (anchors.has(key)) return;
        objectGroup.remove(visual.group);
        disposeObject(visual.group);
        anchorVisuals.delete(key);
        removeLabel(`object:${key}`);
      });
    };

    const syncRays = (now: number) => {
      rays.forEach((ray, key) => {
        let visual = rayVisuals.get(key);
        if (!visual) {
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.02, -0.4), new THREE.Vector3(0, 0.02, -OBSTACLE_RANGE_M)]),
            new THREE.LineDashedMaterial({ color: 0x8fd3ff, dashSize: 0.16, gapSize: 0.12, transparent: true, opacity: 0.85 }),
          );
          line.computeLineDistances();
          const tipGeometry = new THREE.ConeGeometry(0.09, 0.22, 3);
          tipGeometry.rotateX(-Math.PI / 2);
          const tip = new THREE.Mesh(tipGeometry, new THREE.MeshBasicMaterial({ color: 0x8fd3ff, transparent: true, opacity: 0.9 }));
          tip.position.set(0, 0.02, -OBSTACLE_RANGE_M);
          const holder = new THREE.Group();
          holder.add(line, tip);
          rayGroup.add(holder);
          visual = { line, tip };
          rayVisuals.set(key, visual);
        }
        const age = now - ray.seenAt;
        const opacity = Math.max(0.2, 1 - age / OBJECT_FRESH_MS);
        (visual.line.material as THREE.LineDashedMaterial).opacity = 0.85 * opacity;
        (visual.tip.material as THREE.MeshBasicMaterial).opacity = 0.9 * opacity;
        const label = ensureLabel(`ray:${key}`, 'ray', () => scratchLabel.copy(onFloor(OBSTACLE_RANGE_M * 0.86, ray.shown, 0.02)));
        label.priority = 70 + Math.abs(ray.angle);
        const name = ray.label.charAt(0).toUpperCase() + ray.label.slice(1);
        setLabel(label, [{ text: name, color: '#DDF0FB' }, { text: '· no distance', color: '#8FD3FF' }],
          'font-size:10px;padding:2px 6px;border-radius:7px;background:rgba(6,14,18,.86);border:1px dashed rgba(143,211,255,.45)');
      });
      rayVisuals.forEach((visual, key) => {
        if (rays.has(key)) return;
        const holder = visual.line.parent!;
        rayGroup.remove(holder);
        disposeObject(holder);
        rayVisuals.delete(key);
        removeLabel(`ray:${key}`);
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

    /** A remembered object of this label in about this direction, if any. */
    const matchBearing = (key: string | null, label: string, angle: number): Anchor | null => {
      if (key && anchors.has(key)) return anchors.get(key)!;
      let best: Anchor | null = null;
      let bestDelta = (BEARING_MATCH_DEG * Math.PI) / 180;
      for (const anchor of anchors.values()) {
        if (anchor.label !== label || anchor.seenTick === tickCounter) continue;
        const delta = Math.abs(bearingOf(anchor) - angle);
        if (delta < bestDelta) {
          bestDelta = delta;
          best = anchor;
        }
      }
      return best;
    };

    const integrateObjects = (now: number) => {
      let depthObjects = 0;
      detectionsRef.current.slice(0, 10).forEach((item) => {
        const center = item.spatial?.center_m;
        const validCenter = Array.isArray(center) && center.length >= 3 && center.slice(0, 3).every(Number.isFinite);
        const depth = typeof item.distance_m === 'number' && Number.isFinite(item.distance_m) && item.distance_m > 0 ? item.distance_m : null;
        const nx = ((item.x ?? 0.5) + (item.width ?? 0) / 2 - 0.5) * 2;
        const trackKey = item.track_id != null ? `track:${item.track_id}` : null;
        if (!validCenter && depth == null) {
          // No distance: never invent one. Draw the direction only, from where
          // the box sits in the camera's field of view.
          const angle = Math.atan(Math.max(-1, Math.min(1, nx)) * Math.tan(FOV_RAD / 2));
          const remembered = matchBearing(trackKey, item.label, angle);
          if (remembered) {
            remembered.bearingSeenAt = now;
            remembered.seenTick = tickCounter;
            return;
          }
          const key = trackKey ?? `${item.label}:${Math.round((angle * 180) / Math.PI / 10)}`;
          const previous = rays.get(key);
          if (previous) {
            previous.angle = angle;
            previous.seenAt = now;
          } else {
            rays.set(key, { label: item.label, angle, shown: angle, seenAt: now });
          }
          return;
        }
        depthObjects += 1;
        const ny = (0.5 - ((item.y ?? 0.5) + (item.height ?? 0) / 2)) * 2;
        const point = validCenter
          ? new THREE.Vector3(center![0], EYE_HEIGHT_M - center![1], -Math.abs(center![2]))
          : new THREE.Vector3(nx * depth! * 0.55, EYE_HEIGHT_M + ny * depth! * 0.42, -depth!);
        const matched = matchAnchor(trackKey, item.label, point);
        const previous = matched ? anchors.get(matched) : undefined;
        if (previous && matched) {
          previous.x = previous.x * 0.35 + point.x * 0.65;
          previous.y = previous.y * 0.35 + point.y * 0.65;
          previous.z = previous.z * 0.35 + point.z * 0.65;
          previous.label = item.label;
          previous.lastSeenAt = now;
          previous.bearingSeenAt = now;
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
            const label = labels.get(`object:${matched}`);
            if (label) {
              labels.delete(`object:${matched}`);
              labels.set(`object:${trackKey}`, label);
            }
          }
        } else {
          anchorCounter += 1;
          anchors.set(trackKey ?? `seen:${item.label}:${anchorCounter}`, {
            label: item.label, x: point.x, y: point.y, z: point.z, sx: point.x, sy: point.y, sz: point.z,
            firstSeenAt: now, lastSeenAt: now, bearingSeenAt: now, seenTick: tickCounter,
          });
        }
      });
      // A distance is only trusted for OBJECT_MEMORY_MS, even if the object stays in view.
      anchors.forEach((anchor, key) => { if (now - anchor.lastSeenAt > OBJECT_MEMORY_MS) anchors.delete(key); });
      if (anchors.size > MAX_OBJECT_ANCHORS) {
        [...anchors.entries()]
          .sort((left, right) => left[1].lastSeenAt - right[1].lastSeenAt)
          .slice(0, anchors.size - MAX_OBJECT_ANCHORS)
          .forEach(([key]) => anchors.delete(key));
      }
      rays.forEach((ray, key) => { if (now - ray.seenAt > OBJECT_FRESH_MS) rays.delete(key); });
      if (rays.size > MAX_RAYS) {
        [...rays.entries()]
          .sort((left, right) => left[1].seenAt - right[1].seenAt)
          .slice(0, rays.size - MAX_RAYS)
          .forEach(([key]) => rays.delete(key));
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
        // Red when close, blue toward the range limit; fades out as it ages.
        const nearness = 1 - Math.min(1, distanceFromCamera(dot) / OBSTACLE_RANGE_M);
        const freshness = 1 - Math.min(1, (now - dot.seenAt) / OBSTACLE_MEMORY_MS);
        fadeColor.copy(FAR_COLOR).lerp(NEAR_COLOR, nearness).lerp(BLACK, (1 - freshness) * 0.8);
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

    /** Closest measured things right now, by distance along the floor. */
    const findSightings = (now: number) => {
      const found: Array<Sighting & { key: string | null }> = [];
      anchors.forEach((anchor, key) => {
        if (now - anchor.lastSeenAt > NEAREST_WINDOW_MS) return;
        const bearing = bearingOf(anchor);
        found.push({ key, label: anchor.label, distance: Math.hypot(anchor.x, anchor.z), bearing, direction: directionWords(bearing) });
      });
      let closestDot: Dot | null = null;
      let closestDotDistance = Infinity;
      for (const dot of dots.values()) {
        if (now - dot.seenAt > NEAREST_WINDOW_MS) continue;
        const distance = Math.hypot(dot.x, dot.z);
        if (distance < closestDotDistance) {
          closestDotDistance = distance;
          closestDot = dot;
        }
      }
      if (closestDot) {
        const bearing = bearingOf(closestDot);
        found.push({ key: null, label: 'obstacle', distance: closestDotDistance, bearing, direction: directionWords(bearing) });
      }
      found.sort((left, right) => left.distance - right.distance);
      return found;
    };

    /** Directions the camera frame looks blocked in, with no distance. */
    const integrateBlocked = (now: number) => {
      const frame = blockedRef.current;
      if (frame && frame.at !== lastBlockedAt && now - frame.at <= BLOCKED_FRESH_MS) {
        lastBlockedAt = frame.at;
        blockedState = { covered: frame.covered, dark: frame.dark };
        const halfColumn = FOV_RAD / Math.max(1, frame.columns.length) / 2;
        frame.columns.slice(0, MAX_BLOCKED_COLUMNS).forEach((isBlocked, column) => {
          const bearing = Math.atan((frame.centers[column] - 0.5) * 2 * Math.tan(FOV_RAD / 2));
          // A measured surface in this direction already shows where it is.
          let measured = false;
          for (const dot of dots.values()) {
            if (now - dot.seenAt <= NEAREST_WINDOW_MS && Math.abs(bearingOf(dot) - bearing) <= halfColumn + 0.03) {
              measured = true;
              break;
            }
          }
          if (isBlocked && !measured) blockedDirs.set(column, { bearing, seenAt: now });
          else blockedDirs.delete(column);
        });
      }
      blockedDirs.forEach((entry, column) => { if (now - entry.seenAt > BLOCKED_MEMORY_MS) blockedDirs.delete(column); });

      let index = 0;
      let bearingSum = 0;
      blockedDirs.forEach((entry) => {
        bearingSum += entry.bearing;
        const freshness = 1 - Math.min(1, (now - entry.seenAt) / BLOCKED_MEMORY_MS);
        BLOCKED_RADII_M.forEach((radius, ring) => {
          BLOCKED_HEIGHTS_M.forEach((height) => {
            if (index >= blockedCapacity) return;
            const offset = index * 3;
            blockedPositions[offset] = Math.sin(entry.bearing) * radius;
            blockedPositions[offset + 1] = height;
            blockedPositions[offset + 2] = -Math.cos(entry.bearing) * radius;
            // Strongest near the user, fading outward: the distance is unknown.
            blockedColor.copy(BLOCKED_COLOR).lerp(BLACK, Math.min(0.9, (ring / BLOCKED_RADII_M.length) * 0.6 + (1 - freshness) * 0.7));
            blockedColors[offset] = blockedColor.r;
            blockedColors[offset + 1] = blockedColor.g;
            blockedColors[offset + 2] = blockedColor.b;
            index += 1;
          });
        });
      });
      blockedGeometry.setDrawRange(0, index);
      blockedPositionAttribute.needsUpdate = true;
      blockedColorAttribute.needsUpdate = true;

      if (!blockedDirs.size) {
        removeLabel('blocked');
        return null;
      }
      const bearing = bearingSum / blockedDirs.size;
      const cannotSee = blockedState.covered || blockedState.dark;
      const label = ensureLabel('blocked', 'ray', () => blockedLabelPoint.set(Math.sin(bearing) * 2.2, 0.02, -Math.cos(bearing) * 2.2));
      label.priority = 60;
      setLabel(
        label,
        cannotSee
          ? [{ text: blockedState.dark ? 'Camera can\'t see' : 'View covered', color: '#F6E3B8' }, { text: '· too dark or covered', color: '#E0B45C' }]
          : [{ text: 'View blocked', color: '#F6E3B8' }, { text: '· distance unknown', color: '#E0B45C' }],
        'font-size:10px;padding:2px 6px;border-radius:7px;background:rgba(18,14,6,.88);border:1px dashed rgba(224,180,92,.55)',
      );
      return { direction: directionWords(bearing), bearing, covered: blockedState.covered, dark: blockedState.dark };
    };

    const integrate = () => {
      const now = Date.now();
      tickCounter += 1;
      const depthObjects = integrateObjects(now);
      const dotCount = integrateObstacles(now);
      const blockedView = integrateBlocked(now);
      const sightings = findSightings(now);
      nearestKey = sightings[0]?.key ?? null;
      dangerLevel = sightings[0] && sightings[0].distance < DANGER_M ? 1 : 0;
      syncObjects(now);
      syncRays(now);
      let liveObjects = 0;
      let directionOnly = rays.size;
      anchors.forEach((anchor) => {
        if (now - anchor.lastSeenAt <= OBJECT_FRESH_MS) liveObjects += 1;
        else if (now - anchor.bearingSeenAt <= OBJECT_FRESH_MS) directionOnly += 1;
      });
      integrationCount += 1;
      const elapsed = performanceNow() - integrationStarted;
      if (elapsed >= 1000) {
        measuredMapHz = Math.round(integrationCount * 1000 / elapsed * 10) / 10;
        integrationCount = 0;
        integrationStarted = performanceNow();
      }
      const inView = [...rays.values()].sort((left, right) => left.angle - right.angle)
        .map((ray) => `${ray.label} ${directionWords(ray.angle)}`);
      setStats((previous) => ({
        ...previous,
        liveObjects,
        rememberedObjects: anchors.size - liveObjects - (directionOnly - rays.size),
        directionOnly,
        depthObjects,
        dots: dotCount,
        mapHz: measuredMapHz,
        nearest: sightings[0] ?? null,
        others: sightings.slice(1, 3),
        inView: inView.slice(0, 3),
        blockedView,
      }));
    };

    // ------------------------------------------------------------------ camera
    const pivot = new THREE.Vector3(0, 0.4, -2.0);
    let orbitYaw = 0.5;
    let orbitPitch = 0.62;
    let orbitDistance = 6.4;
    const desiredPosition = new THREE.Vector3();
    const orbitTarget = pivot.clone();

    const updateTopFrustum = () => {
      const aspect = viewWidth / Math.max(1, viewHeight);
      // Fit the whole range ring side to side and from just behind the user to past the far ring.
      const halfHeight = Math.max((OBSTACLE_RANGE_M + 1.3) / 2, (OBSTACLE_RANGE_M + 0.5) / aspect) * topZoom;
      topCamera.left = -halfHeight * aspect;
      topCamera.right = halfHeight * aspect;
      topCamera.top = halfHeight;
      topCamera.bottom = -halfHeight;
      topCamera.position.set(0, 20, TOP_CENTER_Z);
      topCamera.up.set(0, 0, -1);
      topCamera.lookAt(0, 0, TOP_CENTER_Z);
      topCamera.updateProjectionMatrix();
    };
    const updateOrbitTarget = () => {
      desiredPosition.set(
        pivot.x + Math.sin(orbitYaw) * Math.cos(orbitPitch) * orbitDistance,
        pivot.y + Math.sin(orbitPitch) * orbitDistance,
        pivot.z + Math.cos(orbitYaw) * Math.cos(orbitPitch) * orbitDistance,
      );
    };
    updateOrbitTarget();
    orbitCamera.position.copy(desiredPosition);
    orbitCamera.lookAt(orbitTarget);

    let appliedView: ViewMode | null = null;
    const applyView = (mode: ViewMode) => {
      if (appliedView === mode) return;
      appliedView = mode;
      const top = mode === 'top';
      // Orthographic points are sized in pixels; perspective ones in meters.
      dotMaterial.sizeAttenuation = !top;
      dotMaterial.size = top ? 9 * pixelRatio : 0.13;
      dotMaterial.needsUpdate = true;
      blockedMaterial.sizeAttenuation = !top;
      blockedMaterial.size = top ? 4.5 * pixelRatio : 0.08;
      blockedMaterial.needsUpdate = true;
      eye.visible = !top;
      anchorVisuals.forEach(({ stem, marker }) => { stem.visible = !top; marker.visible = !top; });
    };

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
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.08 : 0.93;
      if (viewRef.current === 'top') {
        topZoom = Math.min(1.5, Math.max(0.45, topZoom * factor));
        updateTopFrustum();
      } else {
        orbitDistance = Math.min(12, Math.max(2.5, orbitDistance * factor));
      }
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointercancel', onPointerUp);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });
    resetViewRef.current = () => {
      orbitYaw = 0.5;
      orbitPitch = 0.62;
      orbitDistance = 6.4;
      topZoom = 1;
      updateTopFrustum();
    };

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width < 10 || height < 10) return;
      viewWidth = width;
      viewHeight = height;
      orbitCamera.aspect = width / height;
      orbitCamera.updateProjectionMatrix();
      updateTopFrustum();
      renderer.setSize(width, height, false);
      renderer.domElement.style.width = `${width}px`;
      renderer.domElement.style.height = `${height}px`;
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    const timer = window.setInterval(integrate, MAP_INTERVAL_MS);
    let frame = 0;
    const started = performanceNow();
    const tick = () => {
      frame = requestAnimationFrame(tick);
      // Hidden (mini) panel: keep the map's memory, skip drawing.
      if (container.clientHeight < 10) return;
      const mode = viewRef.current;
      applyView(mode);
      const seconds = (performanceNow() - started) / 1000;

      anchors.forEach((anchor, key) => {
        anchor.sx += (anchor.x - anchor.sx) * 0.2;
        anchor.sy += (anchor.y - anchor.sy) * 0.2;
        anchor.sz += (anchor.z - anchor.sz) * 0.2;
        const visual = anchorVisuals.get(key);
        if (!visual) return;
        visual.group.position.set(anchor.sx, 0, anchor.sz);
        const height = Math.max(0.05, anchor.sy);
        visual.marker.position.y = height;
        visual.stem.scale.y = height;
        visual.stem.visible = visual.marker.visible = mode !== 'top';
        // The nearest measured object pulses so the eye finds it first.
        const pulse = key === nearestKey && !reducedMotion ? 1 + 0.35 * (0.5 + 0.5 * Math.sin(seconds * 5)) : 1;
        visual.halo.scale.set(pulse, 1, pulse);
      });
      rays.forEach((ray, key) => {
        ray.shown += (ray.angle - ray.shown) * 0.2;
        const visual = rayVisuals.get(key);
        if (visual) visual.line.parent!.rotation.y = -ray.shown;
      });
      if (!reducedMotion) sweep.rotation.y = -Math.sin(seconds * 1.6) * (FOV_RAD / 2 - 0.06);
      dangerMaterial.opacity += ((dangerLevel ? 0.55 + 0.35 * Math.sin(seconds * 6) : 0) - dangerMaterial.opacity) * 0.15;
      userHalo.scale.setScalar(reducedMotion ? 1 : 1 + 0.06 * Math.sin(seconds * 2));

      let camera: THREE.Camera = topCamera;
      if (mode === 'orbit') {
        updateOrbitTarget();
        orbitCamera.position.lerp(desiredPosition, 0.2);
        orbitCamera.lookAt(orbitTarget);
        camera = orbitCamera;
      }
      renderer.render(scene, camera);
      placeLabels(camera);

      renderedFrames += 1;
      const now = performanceNow();
      if (now - renderStarted >= 1000) {
        const renderFps = Math.round(renderedFrames * 1000 / (now - renderStarted));
        renderedFrames = 0;
        renderStarted = now;
        setStats((previous) => ({ ...previous, renderFps }));
      }
    };
    tick();
    integrate();

    resetRef.current = () => {
      anchors.clear();
      rays.clear();
      dots.clear();
      syncObjects(Date.now());
      syncRays(Date.now());
      nearestKey = null;
      dangerLevel = 0;
      // Do not re-ingest the frame that was on screen when Clear was pressed.
      lastObstacleSequence = obstaclesRef.current?.sequence ?? lastObstacleSequence;
      dotGeometry.setDrawRange(0, 0);
      blockedDirs.clear();
      lastBlockedAt = blockedRef.current?.at ?? lastBlockedAt;
      blockedGeometry.setDrawRange(0, 0);
      removeLabel('blocked');
      setStats((previous) => ({
        ...previous, liveObjects: 0, rememberedObjects: 0, directionOnly: 0, depthObjects: 0, dots: 0, nearest: null, others: [], inView: [], blockedView: null,
      }));
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
      disposeObject(rayGroup);
      disposeObject(floor);
      dotGeometry.dispose();
      blockedGeometry.dispose();
      blockedMaterial.map?.dispose();
      blockedMaterial.dispose();
      dotMaterial.map?.dispose();
      dotMaterial.dispose();
      renderer.dispose();
      labels.clear();
      container.replaceChildren();
    };
  }, []);

  const panelWidth = size === 'large' ? Math.min(640, window_.width - 24) : Math.min(360, window_.width - 24);
  // Large fills the space between the top bar and the bottom card instead of
  // a fixed height, so it never slides under the camera screen's controls.
  const canvasStyle = size === 'mini' ? styles.canvasHidden : size === 'large' ? styles.canvasFill : styles.canvasSmall;
  const { nearest, others, inView, blockedView } = stats;
  const blockedText = blockedView
    ? blockedView.covered || blockedView.dark
      ? `Camera can't see · ${blockedView.dark ? 'too dark' : 'view covered'}`
      : `View blocked ${blockedView.direction} · distance unknown`
    : '';
  const nearestColor = nearest ? urgencyColor(nearest.distance) : '#7D9686';
  const nearestName = nearest ? (nearest.label === 'obstacle' ? 'Obstacle' : nearest.label.charAt(0).toUpperCase() + nearest.label.slice(1)) : '';
  const secondary = others.length
    ? `then ${others.map((item) => `${item.label} ${item.distance.toFixed(1)} m ${item.direction}`).join(' · ')}`
    : inView.length
      ? `in view, no distance: ${inView.join(' · ')}`
      : nearest && blockedText
        ? blockedText
        : '';
  const spoken = nearest
    ? `Nearest: ${nearestName}, ${nearest.distance.toFixed(1)} meters, ${nearest.direction}. ${secondary}`
    : blockedText
      ? `${blockedText}. ${secondary}`
      : `Nothing measured within ${OBSTACLE_RANGE_M} meters. ${secondary}`;

  return (
    <View style={[styles.panel, { width: panelWidth }, size === 'large' && styles.panelFill]}>
      <View style={styles.header}>
        <View style={styles.titleBlock}>
          <View style={styles.titleRow}>
            <View style={styles.liveDot} />
            <Text style={styles.kicker}>LIVE MAP</Text>
          </View>
          <Text style={styles.subtle} numberOfLines={1}>moves with the camera</Text>
        </View>
        <View style={styles.segment} accessibilityRole="radiogroup" accessibilityLabel="Map view">
          {(['top', 'orbit'] as const).map((mode) => (
            <Pressable
              key={mode}
              onPress={() => setView(mode)}
              style={[styles.segmentButton, view === mode && styles.segmentButtonActive]}
              accessibilityRole="radio"
              accessibilityState={{ selected: view === mode }}
              accessibilityLabel={mode === 'top' ? 'Top-down radar view' : '3D view, drag to rotate'}
            >
              <Text style={[styles.segmentText, view === mode && styles.segmentTextActive]}>{mode === 'top' ? 'Radar' : '3D'}</Text>
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

      <View
        style={[styles.nearestCard, { borderColor: nearest ? `${nearestColor}66` : 'rgba(125,150,134,.25)' }]}
        accessibilityLiveRegion="polite"
        accessibilityLabel={spoken}
      >
        <View style={[styles.arrowBadge, { backgroundColor: nearest ? `${nearestColor}22` : blockedView ? '#E0B45C22' : '#0F2119' }]}>
          <Text style={[styles.arrow, {
            color: nearest ? nearestColor : blockedView ? '#E0B45C' : nearestColor,
            transform: [{ rotate: `${nearest ? (nearest.bearing * 180) / Math.PI : blockedView ? (blockedView.bearing * 180) / Math.PI : 0}deg` }],
          }]}
          >
            {nearest || blockedView ? '↑' : '·'}
          </Text>
        </View>
        <View style={styles.nearestBody}>
          <Text style={styles.nearestLabel}>NEAREST</Text>
          {nearest ? (
            <Text style={styles.nearestText} numberOfLines={1}>
              <Text style={{ color: '#F2FAF4' }}>{nearestName} </Text>
              <Text style={{ color: nearestColor }}>{nearest.distance.toFixed(1)} m</Text>
              <Text style={styles.nearestDirection}> {nearest.direction}</Text>
            </Text>
          ) : blockedView ? (
            <Text style={[styles.nearestText, { color: '#E0B45C' }]} numberOfLines={1}>{blockedText}</Text>
          ) : (
            <Text style={[styles.nearestText, { color: '#B7C9BC' }]} numberOfLines={1}>Nothing measured within {OBSTACLE_RANGE_M} m</Text>
          )}
          {secondary ? <Text style={styles.secondary} numberOfLines={1}>{secondary}</Text> : null}
        </View>
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
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: URGENT.near }]} /><Text style={styles.legendText}>{`< ${DANGER_M} m`}</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: URGENT.mid }]} /><Text style={styles.legendText}>{`< ${CAUTION_M} m`}</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: URGENT.far }]} /><Text style={styles.legendText}>farther</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: URGENT.memory }]} /><Text style={styles.legendText}>seen earlier</Text></View>
            <View style={styles.legendItem}><View style={styles.dash} /><Text style={styles.legendText}>direction only</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, styles.swatchGlow]} /><Text style={styles.legendText}>surface</Text></View>
            <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: '#E0B45C' }]} /><Text style={styles.legendText}>blocked, no distance</Text></View>
          </View>

          <View style={styles.footer}>
            <Text style={styles.counts} numberOfLines={1}>
              {stats.liveObjects} measured · {stats.directionOnly} direction · {stats.rememberedObjects} earlier · {stats.dots} dots
            </Text>
            {view === 'orbit' || size === 'large' ? (
              <Pressable onPress={() => resetViewRef.current()} accessibilityRole="button" accessibilityLabel="Reset the map view">
                <Text style={styles.linkText}>reset</Text>
              </Pressable>
            ) : null}
            <Pressable onPress={() => setShowDetails((value) => !value)} accessibilityRole="button" accessibilityLabel="Show map details">
              <Text style={styles.linkText}>{showDetails ? 'less' : 'details'}</Text>
            </Pressable>
            <Pressable onPress={() => resetRef.current()} style={styles.clear} accessibilityRole="button" accessibilityLabel="Clear the map">
              <Text style={styles.clearText}>Clear</Text>
            </Pressable>
          </View>

          {showDetails && (
            <Text style={styles.note}>
              Detect {performance.capture_fps.toFixed(1)} fps{performance.local_backend ? ` on ${performance.local_backend === 'webgpu' ? 'GPU' : 'CPU'}` : ''} · local {performance.local_detection_ms ?? '—'} ms · API {performance.tracking_latency_ms ?? '—'} ms / {performance.tracking_age_ms ?? '—'} ms ({performance.tracking_status})
              {'\n'}Map {stats.mapHz.toFixed(1)} Hz · render {stats.renderFps} fps · objects with depth {stats.depthObjects}
              {'\n'}Distances come from depth and are kept {OBJECT_MEMORY_MS / 1000}s; surfaces within {OBSTACLE_RANGE_M} m are kept {OBSTACLE_MEMORY_MS / 1000}s (max {MAX_OBSTACLE_DOTS}).
              {'\n'}Objects with no depth show as a dashed direction, never a guessed distance. Scroll to zoom; drag to rotate in 3D.
            </Text>
          )}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    position: 'absolute', right: 0, top: 0, backgroundColor: 'rgba(4, 11, 8, 0.93)',
    borderColor: 'rgba(183, 255, 86, 0.32)', borderWidth: 1, borderRadius: 18, padding: 10, gap: 8, zIndex: 6,
    shadowColor: '#000', shadowOpacity: 0.45, shadowRadius: 18, shadowOffset: { width: 0, height: 8 },
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  titleBlock: { flex: 1 },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  liveDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#B7FF56', shadowColor: '#B7FF56', shadowOpacity: 0.9, shadowRadius: 6 },
  kicker: { color: '#B7FF56', fontSize: 12, fontWeight: '900', letterSpacing: 1.2 },
  subtle: { color: '#6F887A', fontSize: 10, fontWeight: '600', marginTop: 1 },
  segment: { flexDirection: 'row', backgroundColor: '#0E1F17', borderRadius: 10, padding: 2, borderWidth: 1, borderColor: '#17301F' },
  segmentButton: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  segmentButtonActive: { backgroundColor: '#B7FF56' },
  segmentText: { color: '#B7C9BC', fontSize: 11, fontWeight: '800' },
  segmentTextActive: { color: '#07110C' },
  iconButton: { width: 28, height: 28, borderRadius: 9, backgroundColor: '#0E1F17', borderWidth: 1, borderColor: '#17301F', alignItems: 'center', justifyContent: 'center' },
  iconButtonText: { color: '#D5E6D8', fontSize: 14, fontWeight: '800' },
  nearestCard: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: '#0A1812', borderRadius: 12, borderWidth: 1, paddingHorizontal: 10, paddingVertical: 8 },
  arrowBadge: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  arrow: { fontSize: 22, fontWeight: '900', lineHeight: 26 },
  nearestBody: { flex: 1, gap: 1 },
  nearestLabel: { color: '#6F887A', fontSize: 9, fontWeight: '900', letterSpacing: 1.2 },
  nearestText: { fontSize: 16, fontWeight: '900' },
  nearestDirection: { color: '#B7C9BC', fontWeight: '700', fontSize: 13 },
  secondary: { color: '#8DA597', fontSize: 11, fontWeight: '600' },
  canvas: { borderRadius: 14, overflow: 'hidden', backgroundColor: '#060F0B', position: 'relative', borderWidth: 1, borderColor: '#12261B' },
  canvasSmall: { height: 190 },
  canvasFill: { flex: 1, minHeight: 180 },
  canvasHidden: { height: 0, opacity: 0, borderWidth: 0 },
  panelFill: { bottom: 0 },
  legend: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: 10, rowGap: 4 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  swatch: { width: 8, height: 8, borderRadius: 4 },
  swatchGlow: { backgroundColor: '#56C8FF', shadowColor: '#56C8FF', shadowOpacity: 0.9, shadowRadius: 4 },
  dash: { width: 12, height: 0, borderTopWidth: 2, borderStyle: 'dashed', borderColor: '#8FD3FF' },
  legendText: { color: '#A9BDB0', fontSize: 10, fontWeight: '600' },
  footer: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  counts: { color: '#C9DACE', fontSize: 10, fontWeight: '700', flex: 1 },
  linkText: { color: '#8FD3FF', fontSize: 10, fontWeight: '700' },
  clear: { backgroundColor: '#163024', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  clearText: { color: '#E7F6EA', fontSize: 11, fontWeight: '800' },
  note: { color: '#A9BDB0', fontSize: 10, lineHeight: 14 },
});

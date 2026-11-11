import { Ionicons } from '@expo/vector-icons';
import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import * as THREE from 'three';
import type { SightObservation } from '../src/campusMemory';

type Props = {
  zoneId: string;
  observations: SightObservation[];
  selectedPoint: { x: number; y: number } | null;
  onSelectPoint: (point: { x: number; y: number }) => void;
  interactive?: boolean;
  /** Replaces the default instruction line (the landing page explains it differently). */
  description?: string;
  /** Extra pill beside the count, e.g. "EXAMPLE" when the pins are sample data. */
  badge?: string;
  /** Slow camera orbit; pass false when the viewer prefers reduced motion. */
  autoRotate?: boolean;
};

type Point2 = { x: number; y: number };
type ViewMode = 'top' | 'orbit';

const ROOM_W = 6.2;
const ROOM_D = 4.6;
const WALL_H = 1.1;
const PIN_H = 0.55;
const ENTRY: Point2 = { x: 0.5, y: 0.9 };

// The live map's palette (LiveSceneMap.web.tsx), so both maps read the same.
const BACKGROUND = new THREE.Color(0x060f0b);
const FLOOR_CENTER = new THREE.Color(0x10281d);
const CATEGORY_COLOR: Record<string, string> = {
  hazard: '#FF6B4A', obstacle: '#FFD043', sign: '#8FD3FF', room_status: '#B7FF56',
};
const REPORT_COLOR = '#B7FF56';
const NEW_PIN_COLOR = '#8FD3FF';

function placedPins(observations: SightObservation[]) {
  return observations
    .filter((item) => item.room_x != null && item.room_y != null && item.verification_status === 'verified')
    .slice()
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
}

function pinColor(category: string) {
  return CATEGORY_COLOR[category] ?? REPORT_COLOR;
}

function timeAgo(timestamp: string) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(timestamp).getTime()) / 60000));
  if (!Number.isFinite(minutes)) return '';
  if (minutes < 1) return 'now';
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.round(minutes / 60);
  return hours < 24 ? `${hours} h` : `${Math.round(hours / 24)} d`;
}

function toWorld(point: Point2, y = 0) {
  return new THREE.Vector3((point.x - 0.5) * ROOM_W, y, (point.y - 0.5) * ROOM_D);
}

function fromWorld(point: THREE.Vector3) {
  return {
    x: Math.max(0.06, Math.min(0.94, point.x / ROOM_W + 0.5)),
    y: Math.max(0.08, Math.min(0.92, point.z / ROOM_D + 0.5)),
  };
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    (Array.isArray(material) ? material : material ? [material] : []).forEach((item) => item.dispose());
  });
}

/** Room floor that glows at the centre and fades into the background at the walls. */
function floorGlow() {
  const geometry = new THREE.PlaneGeometry(ROOM_W + 1.6, ROOM_D + 1.6, 32, 24);
  const positions = geometry.getAttribute('position');
  const colors = new Float32Array(positions.count * 3);
  const mixed = new THREE.Color();
  for (let index = 0; index < positions.count; index += 1) {
    const r = Math.min(1, Math.hypot(positions.getX(index) / (ROOM_W / 2 + 0.8), positions.getY(index) / (ROOM_D / 2 + 0.8)));
    mixed.copy(BACKGROUND).lerp(FLOOR_CENTER, Math.pow(1 - r, 0.85));
    colors.set([mixed.r, mixed.g, mixed.b], index * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, depthWrite: false }));
  mesh.position.y = -0.002;
  return mesh;
}

/** One-metre floor grid, the room outline and low wall edges. */
function roomLines() {
  const group = new THREE.Group();
  const grid: THREE.Vector3[] = [];
  for (let x = -Math.floor(ROOM_W / 2); x <= ROOM_W / 2; x += 1) grid.push(new THREE.Vector3(x, 0.004, -ROOM_D / 2), new THREE.Vector3(x, 0.004, ROOM_D / 2));
  for (let z = -Math.floor(ROOM_D / 2); z <= ROOM_D / 2; z += 1) grid.push(new THREE.Vector3(-ROOM_W / 2, 0.004, z), new THREE.Vector3(ROOM_W / 2, 0.004, z));
  group.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(grid),
    new THREE.LineBasicMaterial({ color: 0x1a3a2b, transparent: true, opacity: 0.55 }),
  ));
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([sx, sz]) => [sx * ROOM_W / 2, sz * ROOM_D / 2]);
  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(corners.map(([x, z]) => new THREE.Vector3(x, 0.008, z))),
    new THREE.LineBasicMaterial({ color: 0x4f8a64 }),
  ));
  const walls: THREE.Vector3[] = [];
  corners.slice(0, 4).forEach(([x, z], index) => {
    const [nx, nz] = corners[index + 1];
    walls.push(new THREE.Vector3(x, 0, z), new THREE.Vector3(x, WALL_H, z));
    walls.push(new THREE.Vector3(x, WALL_H, z), new THREE.Vector3(nx, WALL_H, nz));
  });
  group.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(walls),
    new THREE.LineBasicMaterial({ color: 0x24453a, transparent: true, opacity: 0.7 }),
  ));
  // The doorway: a bright gap marker at the entry.
  const door = toWorld(ENTRY);
  door.z = ROOM_D / 2;
  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([door.clone().setX(door.x - 0.45).setY(0.012), door.clone().setX(door.x + 0.45).setY(0.012)]),
    new THREE.LineBasicMaterial({ color: 0xb7ff56 }),
  ));
  return group;
}

function flatDisc(radius: number, color: string, opacity = 1) {
  const geometry = new THREE.CircleGeometry(radius, 32);
  geometry.rotateX(-Math.PI / 2);
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
}

function flatRing(inner: number, outer: number, color: string, opacity = 1) {
  const geometry = new THREE.RingGeometry(inner, outer, 40);
  geometry.rotateX(-Math.PI / 2);
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide }));
}

const LABEL_STYLE = [
  'position:absolute', 'left:0', 'top:0', 'white-space:nowrap', 'pointer-events:none', 'will-change:transform',
  'font-family:system-ui,-apple-system,Segoe UI,sans-serif', 'font-size:11px', 'padding:3px 7px', 'border-radius:7px',
  'background:rgba(6,15,11,.86)', 'transition:opacity 120ms linear',
].join(';');

type MapLabel = { el: HTMLDivElement; world: THREE.Vector3; lift: number; priority: number; width: number; height: number };

export default function RoomSketchMap({
  zoneId, observations, selectedPoint, onSelectPoint, interactive = true, description, badge, autoRotate = true,
}: Props) {
  const host = useRef<HTMLElement | null>(null);
  const rebuildRef = useRef<() => void>(() => {});
  const resetViewRef = useRef<() => void>(() => {});
  const [view, setView] = useState<ViewMode>('orbit');
  const data = useRef({ observations, selectedPoint, onSelectPoint, interactive, autoRotate, view });
  data.current = { observations, selectedPoint, onSelectPoint, interactive, autoRotate, view };

  const placed = placedPins(observations);
  const unplacedCount = observations.length - placed.length;

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = BACKGROUND.clone();
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.touchAction = 'none';
    container.appendChild(renderer.domElement);
    const labelLayer = document.createElement('div');
    labelLayer.style.cssText = 'position:absolute;inset:0;overflow:hidden;pointer-events:none';
    container.appendChild(labelLayer);

    const topCamera = new THREE.OrthographicCamera(-4, 4, 3, -3, 0.1, 60);
    const orbitCamera = new THREE.PerspectiveCamera(46, 1, 0.05, 80);
    let viewWidth = 1;
    let viewHeight = 1;

    const room = new THREE.Group();
    room.add(floorGlow(), roomLines());
    scene.add(room);
    const dynamic = new THREE.Group();
    scene.add(dynamic);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(ROOM_W, ROOM_D), new THREE.MeshBasicMaterial({ visible: false }));
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const labels: MapLabel[] = [];
    const addLabel = (world: THREE.Vector3, lift: number, priority: number, parts: Array<{ text: string; color: string; weight?: number }>, border: string) => {
      const el = document.createElement('div');
      el.style.cssText = `${LABEL_STYLE};border:1px solid ${border};opacity:0`;
      parts.forEach((part, index) => {
        const span = document.createElement('span');
        span.textContent = index ? ` ${part.text}` : part.text;
        span.style.color = part.color;
        span.style.fontWeight = String(part.weight ?? 600);
        el.appendChild(span);
      });
      labelLayer.appendChild(el);
      labels.push({ el, world, lift, priority, width: el.offsetWidth, height: el.offsetHeight });
    };
    const entryLabelPoint = toWorld(ENTRY).setZ(ROOM_D / 2 + 0.35);

    let pulse: THREE.Mesh | null = null;
    const rebuild = () => {
      disposeObject(dynamic);
      dynamic.clear();
      labels.splice(0).forEach((label) => label.el.remove());
      addLabel(entryLabelPoint, 0, 900, [{ text: 'ENTRY', color: '#7FB98F', weight: 800 }], 'transparent');
      const current = data.current;
      const pins = placedPins(current.observations);

      // The walked path: entry, then each report in time order.
      const stops = [ENTRY, ...pins.map((item) => ({ x: item.room_x ?? 0.5, y: item.room_y ?? 0.5 }))];
      if (current.selectedPoint) stops.push(current.selectedPoint);
      if (stops.length > 1) {
        const curve = new THREE.CatmullRomCurve3(stops.map((point) => toWorld(point, 0.014)));
        dynamic.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(curve.getPoints(96)),
          new THREE.LineBasicMaterial({ color: 0xb7ff56, transparent: true, opacity: 0.6 }),
        ));
      }

      pins.forEach((item, index) => {
        const color = pinColor(item.category);
        const base = toWorld({ x: item.room_x ?? 0.5, y: item.room_y ?? 0.5 });
        const pin = new THREE.Group();
        pin.position.copy(base);
        const disc = flatDisc(0.11, color);
        disc.position.y = 0.03;
        const halo = flatRing(0.17, 0.215, color, 0.8);
        halo.position.y = 0.025;
        const stem = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, PIN_H, 0)]),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 }),
        );
        const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.12), new THREE.MeshBasicMaterial({ color }));
        marker.position.y = PIN_H;
        stem.name = 'stem';
        marker.name = 'marker';
        pin.add(halo, disc, stem, marker);
        dynamic.add(pin);
        const age = timeAgo(item.timestamp);
        addLabel(base, PIN_H + 0.16, index, [
          { text: `${index + 1}`, color, weight: 900 },
          { text: item.label, color: '#F2FAF4', weight: 700 },
          ...(age ? [{ text: age, color: '#8DA597' }] : []),
        ], `${color}55`);
      });

      // The person: a heading chevron at the last stop, facing along the path.
      const here = stops[stops.length - 1];
      const before = stops[Math.max(0, stops.length - 2)];
      const facing = toWorld(here).sub(toWorld(before));
      const heading = facing.lengthSq() < 0.0001 ? 0 : Math.atan2(facing.x, -facing.z);
      const chevronShape = new THREE.Shape();
      chevronShape.moveTo(0, 0.3);
      chevronShape.lineTo(0.2, -0.18);
      chevronShape.lineTo(0, -0.07);
      chevronShape.lineTo(-0.2, -0.18);
      chevronShape.closePath();
      const chevronGeometry = new THREE.ShapeGeometry(chevronShape);
      chevronGeometry.rotateX(-Math.PI / 2);
      const user = new THREE.Group();
      user.add(new THREE.Mesh(chevronGeometry, new THREE.MeshBasicMaterial({ color: 0xb7ff56, side: THREE.DoubleSide })));
      const userHalo = flatRing(0.34, 0.38, '#B7FF56', 0.35);
      user.add(userHalo);
      user.position.copy(toWorld(here, 0.02));
      user.rotation.y = -heading;
      dynamic.add(user);

      pulse = null;
      if (current.selectedPoint) {
        const spot = toWorld(current.selectedPoint);
        pulse = flatRing(0.2, 0.26, NEW_PIN_COLOR, 0.9);
        pulse.position.copy(spot).setY(0.03);
        dynamic.add(pulse);
        addLabel(spot, 0.1, -1, [{ text: 'New pin here', color: NEW_PIN_COLOR, weight: 800 }], `${NEW_PIN_COLOR}55`);
      }
    };
    rebuild();
    rebuildRef.current = rebuild;

    // ------------------------------------------------------------------ views
    let orbitYaw = 0.55;
    let orbitPitch = 0.62;
    let orbitDistance = 9.2;
    let topZoom = 1;
    let lastInteraction = -Infinity;
    const updateTopFrustum = () => {
      const aspect = viewWidth / Math.max(1, viewHeight);
      const half = Math.max((ROOM_D + 1.4) / 2, (ROOM_W + 1.0) / 2 / aspect) * topZoom;
      topCamera.left = -half * aspect;
      topCamera.right = half * aspect;
      topCamera.top = half;
      topCamera.bottom = -half;
      topCamera.position.set(0, 20, 0);
      topCamera.up.set(0, 0, -1);
      topCamera.lookAt(0, 0, 0);
      topCamera.updateProjectionMatrix();
    };
    const resize = () => {
      viewWidth = Math.max(container.clientWidth, 280);
      viewHeight = Math.max(container.clientHeight, 240);
      orbitCamera.aspect = viewWidth / viewHeight;
      orbitCamera.updateProjectionMatrix();
      updateTopFrustum();
      renderer.setSize(viewWidth, viewHeight, false);
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resetViewRef.current = () => {
      orbitYaw = 0.55;
      orbitPitch = 0.62;
      orbitDistance = 9.2;
      topZoom = 1;
      updateTopFrustum();
    };

    // Drag rotates the 3D view; a tap (no drag) places a pin when allowed.
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let press: { x: number; y: number; lastX: number; lastY: number; moved: boolean } | null = null;
    const activeCamera = () => (data.current.view === 'top' ? topCamera : orbitCamera);
    const onPointerDown = (event: PointerEvent) => {
      press = { x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY, moved: false };
      renderer.domElement.setPointerCapture?.(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!press) return;
      if (Math.hypot(event.clientX - press.x, event.clientY - press.y) > 6) press.moved = true;
      if (press.moved && data.current.view === 'orbit') {
        orbitYaw -= (event.clientX - press.lastX) * 0.008;
        orbitPitch = Math.min(1.45, Math.max(0.15, orbitPitch + (event.clientY - press.lastY) * 0.008));
        lastInteraction = performance.now();
      }
      press.lastX = event.clientX;
      press.lastY = event.clientY;
    };
    const onPointerUp = (event: PointerEvent) => {
      const tap = press && !press.moved;
      press = null;
      if (!tap || !data.current.interactive) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, activeCamera());
      const hit = raycaster.intersectObject(floor)[0];
      if (hit) data.current.onSelectPoint(fromWorld(hit.point));
    };
    const onPointerCancel = () => { press = null; };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.08 : 0.93;
      if (data.current.view === 'top') {
        topZoom = Math.min(1.8, Math.max(0.5, topZoom * factor));
        updateTopFrustum();
      } else {
        orbitDistance = Math.min(13, Math.max(3.2, orbitDistance * factor));
      }
      lastInteraction = performance.now();
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointercancel', onPointerCancel);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

    // ----------------------------------------------------------------- labels
    const scratch = new THREE.Vector3();
    const placeLabels = (camera: THREE.Camera, top: boolean) => {
      const taken: Array<[number, number, number, number]> = [];
      const fits = (left: number, topPx: number, w: number, h: number) => {
        if (left < 2 || topPx < 2 || left + w > viewWidth - 2 || topPx + h > viewHeight - 2) return false;
        return taken.every(([l, t, r, b]) => left + w + 3 <= l || left >= r + 3 || topPx + h + 2 <= t || topPx >= b + 2);
      };
      labels.slice().sort((a, b) => a.priority - b.priority).forEach((label) => {
        if (!label.width) {
          label.width = label.el.offsetWidth;
          label.height = label.el.offsetHeight;
        }
        scratch.copy(label.world);
        if (!top) scratch.y += label.lift;
        scratch.project(camera);
        let spot: [number, number] | null = null;
        if (scratch.z >= -1 && scratch.z <= 1) {
          const px = ((scratch.x + 1) / 2) * viewWidth;
          const py = ((1 - scratch.y) / 2) * viewHeight;
          const { width: w, height: h } = label;
          const candidates: Array<[number, number]> = label.priority >= 900
            ? [[px - w / 2, py - h / 2]]
            : [[px - w / 2, py - h - 10], [px + 12, py - h / 2], [px - w - 12, py - h / 2], [px - w / 2, py + 10]];
          spot = candidates.find(([left, t]) => fits(left, t, w, h)) ?? null;
          if (spot) taken.push([spot[0], spot[1], spot[0] + w, spot[1] + h]);
        }
        if (spot) {
          label.el.style.transform = `translate3d(${Math.round(spot[0])}px, ${Math.round(spot[1])}px, 0)`;
          label.el.style.opacity = '1';
        } else {
          label.el.style.opacity = '0';
        }
      });
    };

    // Skip drawing while the map is scrolled off-screen.
    let onScreen = true;
    const visibility = typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; })
      : null;
    visibility?.observe(container);

    let frame = 0;
    let previous = performance.now();
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const now = performance.now();
      const elapsed = Math.min(100, now - previous);
      previous = now;
      if (!onScreen) return;
      const top = data.current.view === 'top';
      // Auto-orbit pauses for a few seconds after the user drags or zooms.
      if (!top && data.current.autoRotate && now - lastInteraction > 4000) orbitYaw += elapsed * 0.00012;
      dynamic.traverse((child) => {
        if (child.name === 'stem' || child.name === 'marker') child.visible = !top;
      });
      if (pulse) {
        const wave = data.current.autoRotate ? (Math.sin(now / 320) + 1) / 2 : 0.5;
        pulse.scale.setScalar(1 + wave * 0.35);
        (pulse.material as THREE.MeshBasicMaterial).opacity = 0.95 - wave * 0.55;
      }
      let camera: THREE.Camera = topCamera;
      if (!top) {
        orbitCamera.position.set(
          Math.sin(orbitYaw) * Math.cos(orbitPitch) * orbitDistance,
          Math.sin(orbitPitch) * orbitDistance,
          Math.cos(orbitYaw) * Math.cos(orbitPitch) * orbitDistance,
        );
        orbitCamera.lookAt(0, 0.25, 0);
        camera = orbitCamera;
      }
      renderer.render(scene, camera);
      placeLabels(camera, top);
    };
    tick();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      visibility?.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('pointercancel', onPointerCancel);
      renderer.domElement.removeEventListener('wheel', onWheel);
      disposeObject(dynamic);
      disposeObject(room);
      renderer.dispose();
      container.replaceChildren();
    };
  }, [zoneId]);

  useEffect(() => {
    rebuildRef.current();
  }, [observations, selectedPoint, interactive]);

  const hint = [interactive ? 'tap the floor to place a pin' : '', view === 'orbit' ? 'drag to rotate' : '', 'scroll to zoom']
    .filter(Boolean).join(' · ');

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.headingIcon}>
          <Ionicons name="cube-outline" size={17} color="#07110C" />
        </View>
        <View style={styles.headingCopy}>
          <View style={styles.titleRow}>
            <View style={styles.liveDot} />
            <Text style={styles.kicker}>3D SPATIAL MAP</Text>
          </View>
          <Text style={styles.title} numberOfLines={1}>{zoneId}</Text>
        </View>
        {badge ? (
          <View style={styles.badgePill}>
            <Text style={styles.badgeText}>{badge}</Text>
          </View>
        ) : null}
        <View style={styles.segment} accessibilityRole="radiogroup" accessibilityLabel="Map view">
          {(['top', 'orbit'] as const).map((mode) => (
            <Pressable
              key={mode}
              onPress={() => setView(mode)}
              style={[styles.segmentButton, view === mode && styles.segmentButtonActive]}
              accessibilityRole="radio"
              accessibilityState={{ selected: view === mode }}
              accessibilityLabel={mode === 'top' ? 'Top-down view of the room' : '3D view, drag to rotate'}
            >
              <Text style={[styles.segmentText, view === mode && styles.segmentTextActive]}>{mode === 'top' ? 'Top' : '3D'}</Text>
            </Pressable>
          ))}
        </View>
      </View>
      <Text style={styles.instruction}>
        {description ?? 'Shared pins stay on this map. The live room map is on the camera screen and fills in as the camera moves.'}
      </Text>
      <View style={styles.viewport}>
        <View
          ref={(node) => {
            const candidate = node as unknown as HTMLElement | { _nativeNode?: HTMLElement } | null;
            host.current = candidate && 'appendChild' in candidate
              ? candidate
              : candidate?._nativeNode ?? null;
          }}
          style={styles.canvasHost}
        />
      </View>
      <View style={styles.legend}>
        <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: CATEGORY_COLOR.hazard }]} /><Text style={styles.legendText}>hazard</Text></View>
        <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: CATEGORY_COLOR.obstacle }]} /><Text style={styles.legendText}>obstacle</Text></View>
        <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: CATEGORY_COLOR.sign }]} /><Text style={styles.legendText}>sign</Text></View>
        <View style={styles.legendItem}><View style={[styles.swatch, { backgroundColor: REPORT_COLOR }]} /><Text style={styles.legendText}>other report</Text></View>
        <View style={styles.legendItem}><View style={styles.pathLine} /><Text style={styles.legendText}>path walked</Text></View>
      </View>
      <View style={styles.footer}>
        <Text style={styles.counts} numberOfLines={1}>{placed.length} {placed.length === 1 ? 'pin' : 'pins'} · {hint}</Text>
        <Pressable onPress={() => resetViewRef.current()} accessibilityRole="button" accessibilityLabel="Reset the map view">
          <Text style={styles.linkText}>reset</Text>
        </Pressable>
      </View>
      {placed.length > 0 && (
        <View style={styles.pinList}>
          {placed.slice(0, 5).map((item, index) => (
            <View key={item.id} style={styles.pinListRow}>
              <View style={[styles.listNumber, { backgroundColor: pinColor(item.category) }]}><Text style={styles.listNumberText}>{index + 1}</Text></View>
              <View style={styles.listCopy}>
                <Text numberOfLines={1} style={styles.itemName}>{item.label}</Text>
                <Text style={styles.itemMeta}>{item.category.replace(/_/g, ' ')} · {Math.round(item.confidence * 100)}% confidence</Text>
              </View>
              <Text style={styles.itemTime}>{new Date(item.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</Text>
            </View>
          ))}
        </View>
      )}
      {unplacedCount > 0 && (
        <Text style={styles.unplacedNote}>{unplacedCount} recent report{unplacedCount === 1 ? '' : 's'} {unplacedCount === 1 ? 'is' : 'are'} not on the path yet.</Text>
      )}
      <Text style={styles.disclaimer}>Pins sit where students tapped them on this room plan. It is not a metric scan.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: 'rgba(4, 11, 8, 0.93)', borderColor: 'rgba(183, 255, 86, 0.32)', borderWidth: 1, borderRadius: 18, padding: 12, marginTop: 12, gap: 8 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  headingIcon: { width: 34, height: 34, borderRadius: 11, backgroundColor: '#B7FF56', alignItems: 'center', justifyContent: 'center' },
  headingCopy: { flex: 1, gap: 2 },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  liveDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#B7FF56', shadowColor: '#B7FF56', shadowOpacity: 0.9, shadowRadius: 6 },
  kicker: { color: '#B7FF56', fontSize: 11, fontWeight: '900', letterSpacing: 1.2 },
  title: { color: '#F2FAF4', fontSize: 15, fontWeight: '800' },
  badgePill: { borderWidth: 1, borderColor: '#2E4A38', borderRadius: 20, paddingHorizontal: 8, paddingVertical: 5 },
  badgeText: { color: '#A9BDB0', fontSize: 9, fontWeight: '900', letterSpacing: 0.8 },
  segment: { flexDirection: 'row', backgroundColor: '#0E1F17', borderRadius: 10, padding: 2, borderWidth: 1, borderColor: '#17301F' },
  segmentButton: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  segmentButtonActive: { backgroundColor: '#B7FF56' },
  segmentText: { color: '#B7C9BC', fontSize: 11, fontWeight: '800' },
  segmentTextActive: { color: '#07110C' },
  instruction: { color: '#8DA597', fontSize: 11, lineHeight: 15 },
  viewport: { height: 340, borderRadius: 14, overflow: 'hidden', borderWidth: 1, borderColor: '#12261B', backgroundColor: '#060F0B' },
  canvasHost: { flex: 1, position: 'relative' },
  legend: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: 10, rowGap: 4 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  swatch: { width: 8, height: 8, borderRadius: 4 },
  pathLine: { width: 12, height: 2, borderRadius: 1, backgroundColor: '#B7FF56', opacity: 0.7 },
  legendText: { color: '#A9BDB0', fontSize: 10, fontWeight: '600' },
  footer: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  counts: { color: '#C9DACE', fontSize: 10, fontWeight: '700', flex: 1 },
  linkText: { color: '#8FD3FF', fontSize: 10, fontWeight: '700' },
  pinList: { borderTopWidth: 1, borderColor: '#17301F', paddingTop: 3 },
  pinListRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: '#12261B' },
  listNumber: { width: 21, height: 21, borderRadius: 11, alignItems: 'center', justifyContent: 'center' },
  listNumberText: { color: '#07110C', fontSize: 9, fontWeight: '900' },
  listCopy: { flex: 1, gap: 1 },
  itemName: { color: '#F2FAF4', fontSize: 11, fontWeight: '800', textTransform: 'capitalize' },
  itemMeta: { color: '#7D9686', fontSize: 9 },
  itemTime: { color: '#6F887A', fontSize: 9 },
  unplacedNote: { color: '#E0B45C', fontSize: 10 },
  disclaimer: { color: '#6F887A', fontSize: 9, lineHeight: 13 },
});

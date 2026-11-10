import { Ionicons } from '@expo/vector-icons';
import { useEffect, useRef } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import * as THREE from 'three';
import type { SightObservation } from '../src/campusMemory';

type Props = {
  zoneId: string;
  observations: SightObservation[];
  selectedPoint: { x: number; y: number } | null;
  onSelectPoint: (point: { x: number; y: number }) => void;
  interactive?: boolean;
};

type Point2 = { x: number; y: number };

const ROOM_W = 6.2;
const ROOM_D = 4.6;
const ENTRY: Point2 = { x: 0.5, y: 0.9 };

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 16777619);
  }
  return hash >>> 0;
}

function random(seed: number) {
  let state = seed || 1;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function toWorld(point: Point2) {
  return new THREE.Vector3((point.x - 0.5) * ROOM_W, 0.06, (point.y - 0.5) * ROOM_D);
}

function fromWorld(point: THREE.Vector3) {
  return {
    x: Math.max(0.06, Math.min(0.94, point.x / ROOM_W + 0.5)),
    y: Math.max(0.08, Math.min(0.92, point.z / ROOM_D + 0.5)),
  };
}

function pointCloud(zoneId: string) {
  const next = random(hashString(zoneId || 'room'));
  const positions: number[] = [];
  const colors: number[] = [];
  const push = (x: number, y: number, z: number, color: number) => {
    positions.push(x, y, z);
    const rgb = new THREE.Color(color);
    colors.push(rgb.r, rgb.g, rgb.b);
  };
  for (let index = 0; index < 420; index += 1) {
    push((next() - 0.5) * ROOM_W * 0.92, next() * 0.04, (next() - 0.5) * ROOM_D * 0.88, next() > 0.5 ? 0x1e4d73 : 0x16324a);
  }
  for (let index = 0; index < 520; index += 1) {
    const edge = Math.floor(next() * 4);
    const alongX = (next() - 0.5) * ROOM_W;
    const alongZ = (next() - 0.5) * ROOM_D;
    const height = next() * 1.7;
    const spot = edge === 0
      ? [alongX, height, -ROOM_D / 2]
      : edge === 1
        ? [alongX, height, ROOM_D / 2]
        : edge === 2
          ? [-ROOM_W / 2, height, alongZ]
          : [ROOM_W / 2, height, alongZ];
    push(spot[0], spot[1], spot[2], next() > 0.45 ? 0x7ec8ff : 0xf0a35e);
  }
  for (let cluster = 0; cluster < 3; cluster += 1) {
    const originX = (next() - 0.5) * ROOM_W * 0.55;
    const originZ = (next() - 0.5) * ROOM_D * 0.45;
    for (let index = 0; index < 36; index += 1) {
      push(originX + (next() - 0.5) * 0.7, next() * 0.55, originZ + (next() - 0.5) * 0.7, 0x4cc38a);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.045, vertexColors: true }));
}

function labelSprite(text: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) return new THREE.Sprite();
  context.fillStyle = 'rgba(7, 11, 16, 0.86)';
  context.fillRect(8, 10, 240, 44);
  context.fillStyle = '#F7FBFF';
  context.font = '700 28px sans-serif';
  context.fillText(text.slice(0, 16), 18, 42);
  const material = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.35, 0.34, 1);
  return sprite;
}

export default function RoomSketchMap({ zoneId, observations, selectedPoint, onSelectPoint, interactive = true }: Props) {
  const host = useRef<HTMLElement | null>(null);
  const rebuildRef = useRef<() => void>(() => {});
  const data = useRef({ observations, selectedPoint, onSelectPoint, interactive });
  data.current = { observations, selectedPoint, onSelectPoint, interactive };

  const placed = observations
    .filter((item) => item.room_x != null && item.room_y != null && item.verification_status === 'verified')
    .slice()
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  const unplacedCount = observations.length - placed.length;

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b10);
    scene.fog = new THREE.Fog(0x070b10, 7, 16);
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 40);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    scene.add(pointCloud(zoneId));
    const dynamic = new THREE.Group();
    scene.add(dynamic);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(ROOM_W, ROOM_D),
      new THREE.MeshBasicMaterial({ visible: false }),
    );
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const rebuild = () => {
      dynamic.clear();
      const current = data.current;
      const pins = current.observations
        .filter((item) => item.room_x != null && item.room_y != null && item.verification_status === 'verified')
        .slice()
        .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      const stops = [ENTRY, ...pins.map((item) => ({ x: item.room_x ?? 0.5, y: item.room_y ?? 0.5 }))];
      if (current.selectedPoint) stops.push(current.selectedPoint);
      if (stops.length > 1) {
        const curve = new THREE.CatmullRomCurve3(stops.map((point) => toWorld(point).setY(0.08)));
        dynamic.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(curve.getPoints(80)),
          new THREE.LineBasicMaterial({ color: 0x39ffb0 }),
        ));
      }

      pins.forEach((item, index) => {
        const position = toWorld({ x: item.room_x ?? 0.5, y: item.room_y ?? 0.5 });
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.08, 16, 16),
          new THREE.MeshBasicMaterial({ color: 0xf0a35e }),
        );
        marker.position.copy(position);
        dynamic.add(marker);
        const label = labelSprite(`${index + 1} ${item.label}`);
        label.position.copy(position).setY(0.38);
        dynamic.add(label);
      });

      const cameraPoint = current.selectedPoint
        ?? (pins.length ? { x: pins[pins.length - 1].room_x ?? 0.5, y: pins[pins.length - 1].room_y ?? 0.5 } : ENTRY);
      const previous = stops[Math.max(0, stops.length - 2)];
      const here = toWorld(cameraPoint);
      const before = toWorld(previous);
      const facing = here.clone().sub(before);
      facing.y = 0;
      if (facing.lengthSq() < 0.0001) facing.set(0, 0, -1);
      const rig = new THREE.Mesh(
        new THREE.ConeGeometry(0.1, 0.28, 10),
        new THREE.MeshBasicMaterial({ color: 0x39ffb0 }),
      );
      rig.position.copy(here).setY(0.18);
      rig.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), facing.normalize());
      dynamic.add(rig);
    };
    rebuild();
    rebuildRef.current = rebuild;

    const resize = () => {
      const width = Math.max(container.clientWidth, 320);
      const height = Math.max(container.clientHeight, 280);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    let frame = 0;
    const started = performance.now();
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const turn = (performance.now() - started) * 0.00012;
      camera.position.set(Math.sin(turn) * 6.4, 3.15, Math.cos(turn) * 6.4);
      camera.lookAt(0, 0.35, 0);
      renderer.render(scene, camera);
    };
    tick();

    const place = (event: PointerEvent) => {
      if (!data.current.interactive) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(floor)[0];
      if (hit) data.current.onSelectPoint(fromWorld(hit.point));
    };
    renderer.domElement.addEventListener('pointerdown', place);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', place);
      renderer.dispose();
      container.replaceChildren();
    };
  }, [zoneId]);

  useEffect(() => {
    rebuildRef.current();
  }, [observations, selectedPoint, interactive]);

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.headingIcon}>
          <Ionicons name="cube-outline" size={18} color="#07110F" />
        </View>
        <View style={styles.headingCopy}>
          <Text style={styles.kicker}>3D SPATIAL MAP</Text>
          <Text style={styles.title}>{zoneId}</Text>
        </View>
        <View style={styles.countPill}>
          <Text style={styles.countText}>{placed.length} objects</Text>
        </View>
      </View>
      <Text style={styles.instruction}>
        Shared pins stay on this sketch. The live room map is on the camera screen and fills in as the camera moves.
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
      <View style={styles.keyRow}>
        <View style={[styles.keySwatch, { backgroundColor: '#7EC8FF' }]} />
        <Text style={styles.keyText}>Room</Text>
        <View style={[styles.keySwatch, { backgroundColor: '#39FFB0' }]} />
        <Text style={styles.keyText}>Path</Text>
        <View style={[styles.keySwatch, { backgroundColor: '#F0A35E' }]} />
        <Text style={styles.keyText}>Object</Text>
      </View>
      {placed.length > 0 && (
        <View style={styles.pinList}>
          {placed.slice(0, 5).map((item, index) => (
            <View key={item.id} style={styles.pinListRow}>
              <View style={styles.listNumber}><Text style={styles.listNumberText}>{index + 1}</Text></View>
              <View style={styles.listCopy}>
                <Text numberOfLines={1} style={styles.itemName}>{item.label}</Text>
                <Text style={styles.itemMeta}>{item.category} · {Math.round(item.confidence * 100)}% confidence</Text>
              </View>
              <Text style={styles.itemTime}>{new Date(item.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</Text>
            </View>
          ))}
        </View>
      )}
      {unplacedCount > 0 && (
        <Text style={styles.unplacedNote}>{unplacedCount} recent report{unplacedCount === 1 ? '' : 's'} {unplacedCount === 1 ? 'is' : 'are'} not on the path yet.</Text>
      )}
      <Text style={styles.disclaimer}>This is a 3D sketch of the tapped positions. It is not a metric scan from a headset tracker.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: '#05080C', borderColor: '#1C3A4A', borderWidth: 1, borderRadius: 18, padding: 14, marginTop: 12, gap: 8 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  headingIcon: { width: 36, height: 36, borderRadius: 11, backgroundColor: '#7EC8FF', alignItems: 'center', justifyContent: 'center' },
  headingCopy: { flex: 1, gap: 2 },
  kicker: { color: '#7EC8FF', fontSize: 9, fontWeight: '900', letterSpacing: 1.2 },
  title: { color: '#F4F8FB', fontSize: 15, fontWeight: '800' },
  countPill: { backgroundColor: '#10202A', borderRadius: 20, paddingHorizontal: 9, paddingVertical: 6 },
  countText: { color: '#D5E6F0', fontSize: 10, fontWeight: '800' },
  instruction: { color: '#9BB0BE', fontSize: 11, lineHeight: 15 },
  viewport: { height: 320, borderRadius: 14, overflow: 'hidden', borderWidth: 1, borderColor: '#234556', backgroundColor: '#070B10' },
  canvasHost: { flex: 1 },
  keyRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  keySwatch: { width: 8, height: 8, borderRadius: 4 },
  keyText: { color: '#8EA3B0', fontSize: 9, marginRight: 6 },
  pinList: { borderTopWidth: 1, borderColor: '#17303A', paddingTop: 3 },
  pinListRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: '#12242C' },
  listNumber: { width: 21, height: 21, borderRadius: 11, backgroundColor: '#F0A35E', alignItems: 'center', justifyContent: 'center' },
  listNumberText: { color: '#1A1008', fontSize: 9, fontWeight: '900' },
  listCopy: { flex: 1, gap: 1 },
  itemName: { color: '#F4F8FB', fontSize: 11, fontWeight: '800', textTransform: 'capitalize' },
  itemMeta: { color: '#7E97A6', fontSize: 9 },
  itemTime: { color: '#6E8794', fontSize: 9 },
  unplacedNote: { color: '#E2C982', fontSize: 10 },
  disclaimer: { color: '#6D8290', fontSize: 9, lineHeight: 13 },
});

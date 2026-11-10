import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import * as THREE from 'three';
import type { LocalDetection } from '../src/detector.types';

type Props = {
  detections: LocalDetection[];
};

type Anchor = {
  key: string;
  label: string;
  x: number;
  y: number;
  z: number;
  confidence: number;
};

const SAMPLE_W = 160;
const SAMPLE_H = 120;
const FLOW_W = 64;
const FLOW_H = 48;
const MAX_POINTS = 9000;
const EYE_HEIGHT = 1.45;

function findCamera() {
  return Array.from(document.querySelectorAll('video')).find((video) => video.readyState >= 2 && video.videoWidth > 0) ?? null;
}

function luminance(pixels: Uint8ClampedArray, width: number, height: number) {
  const gray = new Float32Array(FLOW_W * FLOW_H);
  for (let y = 0; y < FLOW_H; y += 1) {
    for (let x = 0; x < FLOW_W; x += 1) {
      const sourceX = Math.min(width - 1, Math.floor((x + 0.5) * width / FLOW_W));
      const sourceY = Math.min(height - 1, Math.floor((y + 0.5) * height / FLOW_H));
      const offset = (sourceY * width + sourceX) * 4;
      gray[y * FLOW_W + x] = pixels[offset] * 0.3 + pixels[offset + 1] * 0.59 + pixels[offset + 2] * 0.11;
    }
  }
  return gray;
}

function estimateShift(previous: Float32Array, current: Float32Array) {
  let best = Number.POSITIVE_INFINITY;
  let shiftX = 0;
  let shiftY = 0;
  for (let dy = -4; dy <= 4; dy += 1) {
    for (let dx = -4; dx <= 4; dx += 1) {
      let total = 0;
      let count = 0;
      for (let y = 6; y < FLOW_H - 6; y += 2) {
        for (let x = 6; x < FLOW_W - 6; x += 2) {
          const sampleY = y + dy;
          const sampleX = x + dx;
          if (sampleY < 0 || sampleY >= FLOW_H || sampleX < 0 || sampleX >= FLOW_W) continue;
          total += Math.abs(current[y * FLOW_W + x] - previous[sampleY * FLOW_W + sampleX]);
          count += 1;
        }
      }
      if (!count) continue;
      const score = total / count;
      if (score < best) {
        best = score;
        shiftX = dx;
        shiftY = dy;
      }
    }
  }
  return { shiftX, shiftY, error: best };
}

function cameraAxes(yaw: number) {
  return {
    right: new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw)),
    forward: new THREE.Vector3(Math.sin(yaw), 0, -Math.cos(yaw)),
  };
}

function toWorld(local: THREE.Vector3, position: THREE.Vector3, yaw: number) {
  const axes = cameraAxes(yaw);
  return new THREE.Vector3(
    position.x + axes.right.x * local.x + axes.forward.x * local.z,
    position.y + local.y,
    position.z + axes.right.z * local.x + axes.forward.z * local.z,
  );
}

function unproject(nx: number, ny: number, distance: number | null) {
  if (distance != null) {
    return new THREE.Vector3(nx * distance * 0.55, ny * distance * 0.42, distance);
  }
  const dirX = nx * 0.72;
  const dirY = ny * 0.55;
  if (dirY < -0.05) {
    const travel = -EYE_HEIGHT / dirY;
    if (travel > 0.35 && travel < 7.5) return new THREE.Vector3(dirX * travel, -EYE_HEIGHT, travel);
  }
  return new THREE.Vector3(dirX * 3.2, dirY * 3.2, 3.2);
}

function labelSprite(text: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) return new THREE.Sprite();
  context.fillStyle = 'rgba(8, 16, 12, 0.9)';
  context.fillRect(6, 8, 244, 48);
  context.strokeStyle = '#B7FF56';
  context.strokeRect(6, 8, 244, 48);
  context.fillStyle = '#F4FFE4';
  context.font = '700 26px sans-serif';
  context.fillText(text.slice(0, 16), 16, 42);
  const material = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.95, 0.24, 1);
  return sprite;
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    const materials = Array.isArray(material) ? material : material ? [material] : [];
    materials.forEach((item) => {
      const textured = item as THREE.SpriteMaterial;
      textured.map?.dispose?.();
      item.dispose();
    });
  });
}

export default function LiveSceneMap({ detections }: Props) {
  const host = useRef<HTMLElement | null>(null);
  const detectionsRef = useRef(detections);
  const resetRef = useRef<() => void>(() => {});
  detectionsRef.current = detections;
  const [stats, setStats] = useState({ points: 0, objects: 0 });

  useEffect(() => {
    const container = host.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07110c);
    scene.fog = new THREE.Fog(0x07110c, 8, 18);
    const view = new THREE.PerspectiveCamera(50, 1, 0.05, 40);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    const grid = new THREE.GridHelper(10, 10, 0x1d3a2c, 0x143024);
    grid.position.y = 0;
    scene.add(grid);

    const positions = new Float32Array(MAX_POINTS * 3);
    const colors = new Float32Array(MAX_POINTS * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setDrawRange(0, 0);
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.07, vertexColors: true }));
    scene.add(points);

    const pathGeometry = new THREE.BufferGeometry();
    const path = new THREE.Line(pathGeometry, new THREE.LineBasicMaterial({ color: 0xb7ff56 }));
    scene.add(path);
    const rig = new THREE.Mesh(
      new THREE.ConeGeometry(0.08, 0.22, 4),
      new THREE.MeshBasicMaterial({ color: 0xb7ff56, wireframe: true }),
    );
    scene.add(rig);
    const objects = new THREE.Group();
    scene.add(objects);

    const pose = new THREE.Vector3(0, EYE_HEIGHT, 0);
    let yaw = 0;
    let count = 0;
    let previousGray: Float32Array | null = null;
    const trail: THREE.Vector3[] = [pose.clone().setY(0.04)];
    const anchors = new Map<string, Anchor>();
    const scratch = document.createElement('canvas');
    scratch.width = SAMPLE_W;
    scratch.height = SAMPLE_H;
    const context = scratch.getContext('2d', { willReadFrequently: true });

    const pushPoint = (point: THREE.Vector3, red: number, green: number, blue: number) => {
      const index = count < MAX_POINTS ? count : Math.floor(Math.random() * MAX_POINTS);
      if (count < MAX_POINTS) count += 1;
      positions[index * 3] = point.x;
      positions[index * 3 + 1] = point.y;
      positions[index * 3 + 2] = point.z;
      colors[index * 3] = red;
      colors[index * 3 + 1] = green;
      colors[index * 3 + 2] = blue;
    };

    const rebuildObjects = () => {
      disposeObject(objects);
      objects.clear();
      anchors.forEach((anchor) => {
        const box = new THREE.Mesh(
          new THREE.BoxGeometry(0.28, 0.28, 0.28),
          new THREE.MeshBasicMaterial({ color: 0xb7ff56, wireframe: true }),
        );
        box.position.set(anchor.x, Math.max(0.2, anchor.y), anchor.z);
        objects.add(box);
        const sprite = labelSprite(anchor.label);
        sprite.position.set(anchor.x, box.position.y + 0.34, anchor.z);
        objects.add(sprite);
      });
    };

    const integrate = () => {
      const video = findCamera();
      if (!video || !context) return;
      context.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
      const frame = context.getImageData(0, 0, SAMPLE_W, SAMPLE_H);
      const gray = luminance(frame.data, SAMPLE_W, SAMPLE_H);
      if (previousGray) {
        const shift = estimateShift(previousGray, gray);
        if (shift.error < 28) {
          yaw -= shift.shiftX * 0.022;
          const step = Math.max(0, 1.2 - shift.error) * 0.004;
          const axes = cameraAxes(yaw);
          pose.addScaledVector(axes.forward, step);
        }
      }
      previousGray = gray;

      const seen = detectionsRef.current;
      for (let y = 4; y < SAMPLE_H; y += 5) {
        for (let x = 4; x < SAMPLE_W; x += 5) {
          const offset = (y * SAMPLE_W + x) * 4;
          const nx = (x / SAMPLE_W - 0.5) * 2;
          const ny = (0.5 - y / SAMPLE_H) * 2;
          const hit = seen.find((item) => {
            const u = x / SAMPLE_W;
            const v = y / SAMPLE_H;
            return u >= item.x && u <= item.x + item.width && v >= item.y && v <= item.y + item.height;
          });
          const local = unproject(nx, ny, hit?.distance_m ?? null);
          pushPoint(toWorld(local, pose, yaw), frame.data[offset] / 255, frame.data[offset + 1] / 255, frame.data[offset + 2] / 255);
        }
      }

      seen.slice(0, 8).forEach((item) => {
        const key = String(item.id ?? item.label);
        const local = unproject((item.x + item.width / 2 - 0.5) * 2, (0.5 - (item.y + item.height / 2)) * 2, item.distance_m ?? 2);
        const world = toWorld(local, pose, yaw);
        const existing = anchors.get(key);
        if (!existing) {
          anchors.set(key, { key, label: item.label, x: world.x, y: world.y, z: world.z, confidence: item.confidence });
        } else {
          existing.x = existing.x * 0.65 + world.x * 0.35;
          existing.y = existing.y * 0.65 + world.y * 0.35;
          existing.z = existing.z * 0.65 + world.z * 0.35;
          existing.confidence = item.confidence;
        }
      });
      if (anchors.size > 16) {
        const oldest = anchors.keys().next().value;
        if (oldest) anchors.delete(oldest);
      }

      const last = trail[trail.length - 1];
      if (last.distanceTo(new THREE.Vector3(pose.x, 0.04, pose.z)) > 0.12) {
        trail.push(new THREE.Vector3(pose.x, 0.04, pose.z));
        if (trail.length > 80) trail.shift();
      }
      path.geometry.dispose();
      path.geometry = new THREE.BufferGeometry().setFromPoints(trail);
      const axes = cameraAxes(yaw);
      rig.position.copy(pose);
      rig.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axes.forward);
      geometry.setDrawRange(0, count);
      geometry.attributes.position.needsUpdate = true;
      geometry.attributes.color.needsUpdate = true;
      geometry.computeBoundingSphere();
      rebuildObjects();
      setStats({ points: count, objects: anchors.size });
    };

    resetRef.current = () => {
      positions.fill(0);
      colors.fill(0);
      count = 0;
      yaw = 0;
      pose.set(0, EYE_HEIGHT, 0);
      previousGray = null;
      trail.splice(0, trail.length, pose.clone().setY(0.04));
      anchors.clear();
      geometry.setDrawRange(0, 0);
      disposeObject(objects);
      objects.clear();
      setStats({ points: 0, objects: 0 });
    };

    const resize = () => {
      const width = Math.max(container.clientWidth, 220);
      const height = Math.max(container.clientHeight, 140);
      view.aspect = width / height;
      view.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    view.position.set(0, 2.15, 2.4);
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    const timer = window.setInterval(integrate, 550);
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const axes = cameraAxes(yaw);
      const behind = pose.clone().addScaledVector(axes.forward, -2.4);
      behind.y = 2.15;
      view.position.lerp(behind, 0.08);
      view.lookAt(pose.x + axes.forward.x * 1.6, 0.7, pose.z + axes.forward.z * 1.6);
      renderer.render(scene, view);
    };
    tick();

    return () => {
      window.clearInterval(timer);
      cancelAnimationFrame(frame);
      observer.disconnect();
      disposeObject(objects);
      geometry.dispose();
      (points.material as THREE.Material).dispose();
      path.geometry.dispose();
      (path.material as THREE.Material).dispose();
      rig.geometry.dispose();
      (rig.material as THREE.Material).dispose();
      renderer.dispose();
      container.replaceChildren();
    };
  }, []);

  return (
    <View style={styles.panel}>
      <View style={styles.header}>
        <Text style={styles.kicker}>LIVE 3D MAP</Text>
        <Text style={styles.count}>{stats.points} pts · {stats.objects}</Text>
        <Pressable onPress={() => resetRef.current()} style={styles.clear}>
          <Text style={styles.clearText}>Clear</Text>
        </Pressable>
      </View>
      <View
        ref={(node) => {
          const candidate = node as unknown as HTMLElement | { _nativeNode?: HTMLElement } | null;
          host.current = candidate && 'appendChild' in candidate ? candidate : candidate?._nativeNode ?? null;
        }}
        style={styles.canvas}
      />
      <Text style={styles.note}>Move the camera slowly. The room and objects fill in from this view.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    position: 'absolute',
    right: 0,
    top: 0,
    width: 300,
    backgroundColor: 'rgba(5, 12, 9, 0.88)',
    borderColor: 'rgba(183, 255, 86, 0.45)',
    borderWidth: 1,
    borderRadius: 16,
    padding: 8,
    gap: 6,
    zIndex: 6,
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  kicker: { color: '#B7FF56', fontSize: 10, fontWeight: '900', letterSpacing: 0.8 },
  count: { color: '#D5E6D8', fontSize: 10, fontWeight: '700', flex: 1 },
  clear: { backgroundColor: '#163024', borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  clearText: { color: '#E7F6EA', fontSize: 10, fontWeight: '800' },
  canvas: { height: 168, borderRadius: 10, overflow: 'hidden', backgroundColor: '#07110C' },
  note: { color: '#B7C9BC', fontSize: 10, lineHeight: 13 },
});

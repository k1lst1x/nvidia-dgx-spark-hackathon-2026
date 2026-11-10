import { Ionicons } from '@expo/vector-icons';
import { useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { SightObservation } from '../src/campusMemory';

type Props = {
  zoneId: string;
  observations: SightObservation[];
  selectedPoint: { x: number; y: number } | null;
  onSelectPoint: (point: { x: number; y: number }) => void;
  interactive?: boolean;
};

type Point = { x: number; y: number; color: string; size: number };

const ENTRY = { x: 0.5, y: 0.9 };

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

function roomCloud(zoneId: string): Point[] {
  const next = random(hashString(zoneId || 'room'));
  const points: Point[] = [];
  for (let index = 0; index < 90; index += 1) {
    points.push({
      x: 0.12 + next() * 0.76,
      y: 0.16 + next() * 0.68,
      color: next() > 0.55 ? '#1E4D73' : '#16324A',
      size: 2,
    });
  }
  for (let index = 0; index < 70; index += 1) {
    const edge = Math.floor(next() * 4);
    const along = 0.08 + next() * 0.84;
    const jitter = (next() - 0.5) * 0.035;
    const spot = edge === 0
      ? { x: along, y: 0.1 + jitter }
      : edge === 1
        ? { x: along, y: 0.9 + jitter }
        : edge === 2
          ? { x: 0.08 + jitter, y: along }
          : { x: 0.92 + jitter, y: along };
    points.push({ ...spot, color: next() > 0.4 ? '#7EC8FF' : '#F0A35E', size: 3 });
  }
  for (let cluster = 0; cluster < 3; cluster += 1) {
    const originX = 0.25 + next() * 0.5;
    const originY = 0.28 + next() * 0.4;
    for (let index = 0; index < 14; index += 1) {
      points.push({
        x: originX + (next() - 0.5) * 0.12,
        y: originY + (next() - 0.5) * 0.1,
        color: '#4CC38A',
        size: 2,
      });
    }
  }
  return points;
}

function pathDots(stops: { x: number; y: number }[]) {
  const dots: { x: number; y: number }[] = [];
  for (let index = 0; index < stops.length - 1; index += 1) {
    const start = stops[index];
    const end = stops[index + 1];
    for (let step = 0; step < 14; step += 1) {
      const t = step / 14;
      dots.push({
        x: start.x + (end.x - start.x) * t,
        y: start.y + (end.y - start.y) * t,
      });
    }
  }
  return dots;
}

export default function RoomSketchMap({ zoneId, observations, selectedPoint, onSelectPoint, interactive = true }: Props) {
  const [mapSize, setMapSize] = useState({ width: 0, height: 0 });
  const cloud = useMemo(() => roomCloud(zoneId), [zoneId]);
  const placed = observations
    .filter((item) => item.room_x != null && item.room_y != null && item.verification_status === 'verified')
    .slice()
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  const unplacedCount = observations.length - placed.length;
  const stops = [ENTRY, ...placed.map((item) => ({ x: item.room_x ?? 0.5, y: item.room_y ?? 0.5 }))];
  if (selectedPoint) stops.push(selectedPoint);
  const trail = pathDots(stops);
  const camera = selectedPoint ?? (placed.length ? { x: placed[placed.length - 1].room_x ?? 0.5, y: placed[placed.length - 1].room_y ?? 0.5 } : ENTRY);

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.headingIcon}>
          <Ionicons name="scan-outline" size={18} color="#07110F" />
        </View>
        <View style={styles.headingCopy}>
          <Text style={styles.kicker}>SPATIAL MAP</Text>
          <Text style={styles.title}>{zoneId}</Text>
        </View>
        <View style={styles.countPill}>
          <Text style={styles.countText}>{placed.length} objects</Text>
        </View>
      </View>

      <Text style={styles.instruction}>
        A reconstructed room view: blue points are the space, the green line is the path, and the marker is the camera. Tap to drop the next object.
      </Text>

      <Pressable
        accessibilityRole={interactive ? 'imagebutton' : 'image'}
        accessibilityLabel={interactive ? 'Spatial map. Tap to place the next report on the path.' : 'Spatial map of verified observations.'}
        onLayout={(event) => setMapSize(event.nativeEvent.layout)}
        onPress={(event) => {
          if (!interactive || !mapSize.width || !mapSize.height) return;
          const { locationX, locationY } = event.nativeEvent;
          onSelectPoint({
            x: Math.max(0.06, Math.min(0.94, locationX / mapSize.width)),
            y: Math.max(0.08, Math.min(0.92, locationY / mapSize.height)),
          });
        }}
        style={[styles.map, !interactive && styles.mapReadOnly]}
      >
        {cloud.map((point, index) => (
          <View
            key={`cloud-${index}`}
            pointerEvents="none"
            style={[styles.cloudPoint, {
              left: `${point.x * 100}%`,
              top: `${point.y * 100}%`,
              width: point.size,
              height: point.size,
              backgroundColor: point.color,
            }]}
          />
        ))}
        {trail.map((point, index) => (
          <View
            key={`trail-${index}`}
            pointerEvents="none"
            style={[styles.trailPoint, { left: `${point.x * 100}%`, top: `${point.y * 100}%` }]}
          />
        ))}
        <View pointerEvents="none" style={[styles.entry, { left: `${ENTRY.x * 100}%`, top: `${ENTRY.y * 100}%` }]}>
          <Text style={styles.entryText}>IN</Text>
        </View>
        {placed.map((item, index) => (
          <View
            key={item.id}
            pointerEvents="none"
            style={[styles.objectPin, { left: `${(item.room_x ?? 0.5) * 100}%`, top: `${(item.room_y ?? 0.5) * 100}%` }]}
          >
            <View style={styles.objectDot} />
            <Text numberOfLines={1} style={styles.objectLabel}>{index + 1} {item.label}</Text>
          </View>
        ))}
        <View pointerEvents="none" style={[styles.camera, { left: `${camera.x * 100}%`, top: `${camera.y * 100}%` }]}>
          <Ionicons name="navigate" size={16} color="#F4FFF8" />
        </View>
      </Pressable>

      <View style={styles.keyRow}>
        <View style={[styles.keySwatch, { backgroundColor: '#7EC8FF' }]} />
        <Text style={styles.keyText}>Room</Text>
        <View style={[styles.keySwatch, { backgroundColor: '#39FFB0' }]} />
        <Text style={styles.keyText}>Path</Text>
        <Ionicons name="navigate" size={11} color="#F4FFF8" />
        <Text style={styles.keyText}>Camera</Text>
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
        <Text style={styles.unplacedNote}>{unplacedCount} recent report{unplacedCount === 1 ? '' : 's'} {unplacedCount === 1 ? 'is' : 'are'} not on the path yet. Place {unplacedCount === 1 ? 'it' : 'them'} by tapping the map.</Text>
      )}
      <Text style={styles.disclaimer}>This draws the same kind of picture as an egocentric SLAM demo: a point cloud, a walked path, and a camera marker. Positions are the ones people tap. It is not a metric 3D reconstruction.</Text>
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
  map: { height: 280, borderRadius: 14, borderWidth: 1, borderColor: '#234556', backgroundColor: '#070B10', overflow: 'hidden', position: 'relative' },
  mapReadOnly: { opacity: 0.92 },
  cloudPoint: { position: 'absolute', borderRadius: 2, marginLeft: -1, marginTop: -1 },
  trailPoint: { position: 'absolute', width: 4, height: 4, borderRadius: 2, marginLeft: -2, marginTop: -2, backgroundColor: '#39FFB0' },
  entry: { position: 'absolute', marginLeft: -12, marginTop: -9, backgroundColor: '#10202A', borderColor: '#7EC8FF', borderWidth: 1, borderRadius: 6, paddingHorizontal: 4, paddingVertical: 2 },
  entryText: { color: '#7EC8FF', fontSize: 8, fontWeight: '900' },
  objectPin: { position: 'absolute', marginLeft: -4, marginTop: -4, flexDirection: 'row', alignItems: 'center', gap: 4, maxWidth: 120 },
  objectDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#F0A35E', borderWidth: 1, borderColor: '#FFF6EA' },
  objectLabel: { color: '#F7FBFF', fontSize: 9, fontWeight: '800', backgroundColor: 'rgba(7, 11, 16, 0.78)', paddingHorizontal: 4, paddingVertical: 1, borderRadius: 4, textTransform: 'capitalize' },
  camera: { position: 'absolute', marginLeft: -10, marginTop: -10, width: 22, height: 22, borderRadius: 11, backgroundColor: '#123226', borderWidth: 1, borderColor: '#39FFB0', alignItems: 'center', justifyContent: 'center' },
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

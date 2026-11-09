import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { SightObservation } from '../src/campusMemory';

type Props = {
  zoneId: string;
  observations: SightObservation[];
  selectedPoint: { x: number; y: number } | null;
  onSelectPoint: (point: { x: number; y: number }) => void;
  interactive?: boolean;
};

const GRID_STEPS = [20, 40, 60, 80];

export default function RoomSketchMap({ zoneId, observations, selectedPoint, onSelectPoint, interactive = true }: Props) {
  const [mapSize, setMapSize] = useState({ width: 0, height: 0 });
  const placed = observations.filter(
    (item) => item.room_x != null && item.room_y != null && item.verification_status === 'verified'
  );
  const unplacedCount = observations.length - placed.length;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.headingIcon}>
          <Ionicons name="map-outline" size={18} color="#07110F" />
        </View>
        <View style={styles.headingCopy}>
          <Text style={styles.kicker}>SHARED ROOM SKETCH</Text>
          <Text style={styles.title}>{zoneId}</Text>
        </View>
        <View style={styles.countPill}>
          <Text style={styles.countText}>{placed.length} pins</Text>
        </View>
      </View>

      <Text style={styles.instruction}>Tap the sketch to place the next verified report.</Text>
      <View style={styles.edgeLabelRow}>
        <Text style={styles.edgeLabel}>FAR SIDE</Text>
        <Text style={styles.edgeLabel}>ROOM CENTER</Text>
        <Text style={styles.edgeLabel}>ENTRY SIDE</Text>
      </View>

      <Pressable
        accessibilityRole={interactive ? 'imagebutton' : 'image'}
        accessibilityLabel={interactive ? 'Room sketch. Tap to choose a location for the next report.' : 'Room sketch showing verified observations.'}
        onLayout={(event) => setMapSize(event.nativeEvent.layout)}
        onPress={(event) => {
          if (!interactive || !mapSize.width || !mapSize.height) return;
          const { locationX, locationY } = event.nativeEvent;
          onSelectPoint({
            x: Math.max(0.03, Math.min(0.97, locationX / mapSize.width)),
            y: Math.max(0.03, Math.min(0.97, locationY / mapSize.height)),
          });
        }}
        style={[styles.map, !interactive && styles.mapReadOnly]}
      >
        <View pointerEvents="none" style={styles.roomOutline} />
        {GRID_STEPS.map((step) => (
          <View key={`v${step}`} pointerEvents="none" style={[styles.gridLine, styles.verticalLine, { left: `${step}%` }]} />
        ))}
        {GRID_STEPS.map((step) => (
          <View key={`h${step}`} pointerEvents="none" style={[styles.gridLine, styles.horizontalLine, { top: `${step}%` }]} />
        ))}
        <View pointerEvents="none" style={styles.entryMarker}>
          <Ionicons name="enter-outline" size={15} color="#9AB2A6" />
        </View>
        {placed.map((item, index) => (
          <View
            key={item.id}
            pointerEvents="none"
            style={[
              styles.pin,
              { left: `${(item.room_x ?? 0.5) * 100}%`, top: `${(item.room_y ?? 0.5) * 100}%` },
            ]}
          >
            <Text style={styles.pinNumber}>{index + 1}</Text>
          </View>
        ))}
        {selectedPoint && (
          <View
            pointerEvents="none"
            style={[
              styles.selectedPin,
              { left: `${selectedPoint.x * 100}%`, top: `${selectedPoint.y * 100}%` },
            ]}
          >
            <View style={styles.selectedPinDot} />
          </View>
        )}
        {!placed.length && !selectedPoint && (
          <View pointerEvents="none" style={styles.emptyPrompt}>
            <Ionicons name="add-circle-outline" size={23} color="#B7FF56" />
            <Text style={styles.emptyPromptText}>Tap to place your first pin</Text>
          </View>
        )}
      </Pressable>

      <View style={styles.keyRow}>
        <View style={styles.keyDot} />
        <Text style={styles.keyText}>Verified, user-placed observation</Text>
        {selectedPoint && <Text style={styles.selectedText}>New pin selected</Text>}
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
        <Text style={styles.unplacedNote}>{unplacedCount} recent report{unplacedCount === 1 ? '' : 's'} not placed on this sketch.</Text>
      )}
      <Text style={styles.disclaimer}>Not to scale. Pins are placed by people; this is not an automatic SLAM map.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: '#091A15', borderColor: '#345C35', borderWidth: 1, borderRadius: 18, padding: 14, marginTop: 12, gap: 8 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  headingIcon: { width: 36, height: 36, borderRadius: 11, backgroundColor: '#B7FF56', alignItems: 'center', justifyContent: 'center' },
  headingCopy: { flex: 1, gap: 2 },
  kicker: { color: '#B7FF56', fontSize: 9, fontWeight: '900', letterSpacing: 1.2 },
  title: { color: '#F3F8F5', fontSize: 15, fontWeight: '800' },
  countPill: { backgroundColor: '#142B21', borderRadius: 20, paddingHorizontal: 9, paddingVertical: 6 },
  countText: { color: '#D2E1D9', fontSize: 10, fontWeight: '800' },
  instruction: { color: '#AFC2B8', fontSize: 11, lineHeight: 15 },
  edgeLabelRow: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 3 },
  edgeLabel: { color: '#6F897C', fontSize: 8, fontWeight: '900', letterSpacing: 0.8 },
  map: { height: 220, borderRadius: 14, borderWidth: 1, borderColor: '#42604D', backgroundColor: '#0D261D', overflow: 'hidden', position: 'relative' },
  mapReadOnly: { opacity: 0.9 },
  roomOutline: { position: 'absolute', left: '4%', top: '6%', width: '92%', height: '84%', borderColor: '#65816B', borderWidth: 1, borderRadius: 10, backgroundColor: 'rgba(22, 55, 39, 0.24)' },
  gridLine: { position: 'absolute', backgroundColor: 'rgba(140, 176, 145, 0.16)' },
  verticalLine: { top: '6%', bottom: '10%', width: 1 },
  horizontalLine: { left: '4%', right: '4%', height: 1 },
  entryMarker: { position: 'absolute', left: '46%', bottom: '0%', width: '10%', height: '10%', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0D261D' },
  pin: { position: 'absolute', marginLeft: -13, marginTop: -13, width: 26, height: 26, borderRadius: 13, backgroundColor: '#B7FF56', borderWidth: 2, borderColor: '#07110F', alignItems: 'center', justifyContent: 'center', shadowColor: '#B7FF56', shadowOpacity: 0.38, shadowRadius: 7, elevation: 4 },
  pinNumber: { color: '#07110F', fontSize: 10, fontWeight: '900' },
  selectedPin: { position: 'absolute', marginLeft: -12, marginTop: -12, width: 24, height: 24, borderRadius: 12, borderWidth: 2, borderColor: '#FFF2A8', backgroundColor: 'rgba(255, 242, 168, 0.35)', alignItems: 'center', justifyContent: 'center' },
  selectedPinDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#FFF2A8' },
  emptyPrompt: { position: 'absolute', alignSelf: 'center', top: '39%', alignItems: 'center', gap: 6 },
  emptyPromptText: { color: '#C0D2C7', fontSize: 11, fontWeight: '700' },
  keyRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  keyDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: '#B7FF56' },
  keyText: { color: '#9FB5A9', fontSize: 9, flex: 1 },
  selectedText: { color: '#FFF2A8', fontSize: 9, fontWeight: '800' },
  pinList: { borderTopWidth: 1, borderColor: '#234135', paddingTop: 3 },
  pinListRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: '#183329' },
  listNumber: { width: 21, height: 21, borderRadius: 11, backgroundColor: '#B7FF56', alignItems: 'center', justifyContent: 'center' },
  listNumberText: { color: '#07110F', fontSize: 9, fontWeight: '900' },
  listCopy: { flex: 1, gap: 1 },
  itemName: { color: '#EFF7F2', fontSize: 11, fontWeight: '800', textTransform: 'capitalize' },
  itemMeta: { color: '#829C8E', fontSize: 9 },
  itemTime: { color: '#789187', fontSize: 9 },
  unplacedNote: { color: '#E2C982', fontSize: 10 },
  disclaimer: { color: '#70877B', fontSize: 9, lineHeight: 13 },
});

import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useState } from 'react';
import * as Speech from 'expo-speech';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import RoomSketchMap from './RoomSketchMap';
import {
  currentSightDevice,
  getSightChanges,
  getSightDashboard,
  removeSightObservation,
  searchPersonalRecall,
  submitSightObservation,
  type SightChanges,
  type SightDashboard as SightDashboardData,
  type SightObservation,
} from '../src/campusMemory';

type Props = { zoneId: string };
type Scope = 'shared' | 'personal';

export default function SharedSightDashboard({ zoneId }: Props) {
  const [dashboard, setDashboard] = useState<SightDashboardData | null>(null);
  const [changes, setChanges] = useState<SightChanges | null>(null);
  const [recall, setRecall] = useState<SightObservation[]>([]);
  const [query, setQuery] = useState('');
  const [label, setLabel] = useState('');
  const [description, setDescription] = useState('');
  const [confidence, setConfidence] = useState('0.90');
  const [category, setCategory] = useState('object');
  const [scope, setScope] = useState<Scope>('shared');
  const [verified, setVerified] = useState(false);
  const [notice, setNotice] = useState('');
  const [loading, setLoading] = useState(false);
  const [accessKey, setAccessKey] = useState('');
  const [roomPoint, setRoomPoint] = useState<{ x: number; y: number } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [next, delta] = await Promise.all([getSightDashboard(zoneId), getSightChanges(zoneId)]);
      setDashboard(next);
      setChanges(delta);
      setNotice('');
    } catch (error) {
      setNotice((error as Error).message);
    }
  }, [zoneId]);

  useEffect(() => {
    setRoomPoint(null);
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        setAccessKey(window.localStorage.getItem('guidelens_shared_sight_key') || '');
      }
    } catch {}
    void refresh();
    const timer = setInterval(() => void refresh(), 15000);
    return () => clearInterval(timer);
  }, [refresh]);

  const search = async () => {
    try {
      const found = await searchPersonalRecall(query);
      setRecall(found);
      setNotice('');
      // Answer out loud with place and time, for users who cannot read the list.
      if (query.trim()) {
        const best = found[0];
        if (best) {
          const minutes = Math.max(0, Math.round((Date.now() - new Date(best.timestamp).getTime()) / 60000));
          const when = minutes < 1 ? 'just now' : minutes < 60 ? `${minutes} minute${minutes === 1 ? '' : 's'} ago`
            : `${Math.round(minutes / 60)} hour${Math.round(minutes / 60) === 1 ? '' : 's'} ago`;
          const detail = best.description && best.description !== best.label ? ` ${best.description}.` : '';
          Speech.speak(`Your ${best.label}: ${best.zone_id}, ${when}.${detail}`, { rate: 0.95 });
        } else {
          Speech.speak(`I could not find that in your saved notes.`, { rate: 0.95 });
        }
      }
    } catch (error) { setNotice((error as Error).message); }
  };

  const submit = async () => {
    if (!label.trim()) return;
    if (scope === 'shared' && !verified) {
      setNotice('Confirm that you checked the report before sharing it.');
      return;
    }
    if (scope === 'shared' && !roomPoint) {
      setNotice('Tap the room sketch to place this shared report.');
      return;
    }
    setLoading(true);
    try {
      await submitSightObservation({
        zone_id: zoneId,
        category,
        label: label.trim(),
        description: description.trim(),
        confidence: Number(confidence),
        privacy_scope: scope,
        verified,
        ...(scope === 'shared' && roomPoint ? { room_x: roomPoint.x, room_y: roomPoint.y } : {}),
      });
      setLabel('');
      setDescription('');
      setRoomPoint(null);
      setNotice(scope === 'shared' ? 'Verified report shared with authorized devices.' : 'Saved to Personal Recall on this session.');
      await refresh();
      if (scope === 'personal') setRecall(await searchPersonalRecall(query));
    } catch (error) { setNotice((error as Error).message); }
    finally { setLoading(false); }
  };

  const remove = async (item: SightObservation) => {
    try {
      await removeSightObservation(item.id);
      await refresh();
      if (item.privacy_scope === 'personal') setRecall(await searchPersonalRecall(query));
      setNotice('Your report was removed.');
    } catch (error) { setNotice((error as Error).message); }
  };

  const renderObservation = (item: SightObservation) => (
    <View key={item.id} style={styles.observation}>
      <View style={styles.observationBody}>
        <Text style={styles.observationTitle}>{item.label}</Text>
        <Text style={styles.observationMeta}>
          {item.category} · {Math.round(item.confidence * 100)}% · {new Date(item.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })} · {item.privacy_scope}
        </Text>
        {!!item.description && <Text style={styles.observationDescription}>{item.description}</Text>}
      </View>
      {item.source_device === currentSightDevice() && (
        <Pressable onPress={() => void remove(item)} accessibilityLabel={`Remove ${item.label}`} style={styles.removeButton}>
          <Ionicons name="trash-outline" size={16} color="#FF7865" />
        </Pressable>
      )}
    </View>
  );

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View>
          <Text style={styles.kicker}>SHARED SIGHT · {zoneId.toUpperCase()}</Text>
          <Text style={styles.title}>Recent verified observations</Text>
        </View>
        <Pressable onPress={() => void refresh()} style={styles.refresh} accessibilityLabel="Refresh Shared Sight">
          <Ionicons name="refresh" size={17} color="#B7FF56" />
        </Pressable>
      </View>
      <View style={styles.metrics}>
        <Text style={styles.metric}>{dashboard?.connected_devices ?? 0} {dashboard?.connected_devices === 1 ? 'phone' : 'phones'} active</Text>
        <Text style={styles.metric}>{dashboard?.tracking_latency_ms == null ? 'latency —' : `tracking ${dashboard.tracking_latency_ms} ms avg`}</Text>
        <Text style={styles.metric}>{dashboard?.tracking_requests_last_5m ?? 0} scans / 5m</Text>
        <Text style={styles.metric}>
          {dashboard?.gpu?.available && dashboard.gpu.gpus[0]
            ? `Spark GPU ${dashboard.gpu.gpus[0].utilization_pct ?? '—'}%`
            : 'GPU not reported'}
        </Text>
      </View>
      <Text style={styles.deviceNote}>This device: {currentSightDevice()}. Configure SHARED_SIGHT_DEVICE_IDS on the server to restrict access.</Text>
      <RoomSketchMap
        zoneId={zoneId}
        observations={dashboard?.recent_observations ?? []}
        selectedPoint={scope === 'shared' ? roomPoint : null}
        onSelectPoint={setRoomPoint}
        interactive={scope === 'shared'}
      />
      <TextInput
        value={accessKey}
        onChangeText={(value) => {
          setAccessKey(value);
          try {
            if (typeof window !== 'undefined' && window.localStorage) window.localStorage.setItem('guidelens_shared_sight_key', value);
          } catch {}
        }}
        placeholder="Shared Sight access key (if your server requires one)"
        placeholderTextColor="#789187"
        secureTextEntry
        style={styles.input}
        autoCapitalize="none"
        autoCorrect={false}
      />
      {changes && <Text style={styles.changeText}>{changes.summary}</Text>}
      {dashboard?.recent_observations.length
        ? dashboard.recent_observations.slice(0, 5).map(renderObservation)
        : <Text style={styles.empty}>No verified reports for this zone yet.</Text>}

      <View style={styles.divider} />
      <Text style={styles.sectionTitle}>Add a report or save a personal note</Text>
      <View style={styles.chips}>
        {['object', 'hazard', 'obstacle', 'room_status'].map((value) => (
          <Pressable key={value} onPress={() => setCategory(value)} style={[styles.chip, category === value && styles.chipActive]}>
            <Text style={[styles.chipText, category === value && styles.chipTextActive]}>{value.replace('_', ' ')}</Text>
          </Pressable>
        ))}
      </View>
      <TextInput value={label} onChangeText={setLabel} placeholder="What did you observe?" placeholderTextColor="#789187" style={styles.input} />
      <TextInput value={description} onChangeText={setDescription} placeholder="Optional detail (no names or identifying details)" placeholderTextColor="#789187" style={styles.input} />
      <TextInput value={confidence} onChangeText={setConfidence} keyboardType="decimal-pad" placeholder="Confidence 0.65–1.0" placeholderTextColor="#789187" style={styles.input} />
      <View style={styles.chips}>
        {(['shared', 'personal'] as const).map((value) => (
          <Pressable key={value} onPress={() => setScope(value)} style={[styles.chip, scope === value && styles.chipActive]}>
            <Text style={[styles.chipText, scope === value && styles.chipTextActive]}>{value === 'shared' ? 'Share with campus' : 'Personal Recall'}</Text>
          </Pressable>
        ))}
      </View>
      {scope === 'shared' && (
        <Pressable onPress={() => setVerified((value) => !value)} style={styles.verifyRow} accessibilityRole="checkbox" accessibilityState={{ checked: verified }}>
          <Ionicons name={verified ? 'checkbox' : 'square-outline'} size={19} color="#B7FF56" />
          <Text style={styles.verifyText}>I checked this report and confirm it is accurate.</Text>
        </Pressable>
      )}
      <Pressable onPress={() => void submit()} disabled={loading || !label.trim()} style={[styles.submit, (loading || !label.trim()) && styles.disabled]}>
        <Text style={styles.submitText}>{loading ? 'Saving…' : scope === 'shared' ? 'Share verified report' : 'Save privately'}</Text>
      </Pressable>
      {!!notice && <Text accessibilityLiveRegion="polite" style={styles.notice}>{notice}</Text>}

      <View style={styles.divider} />
      <Text style={styles.sectionTitle}>Personal Recall · this device</Text>
      <View style={styles.searchRow}>
        <TextInput value={query} onChangeText={setQuery} placeholder="Search your saved observations" placeholderTextColor="#789187" style={[styles.input, styles.searchInput]} onSubmitEditing={() => void search()} />
        <Pressable onPress={() => void search()} style={styles.searchButton}><Ionicons name="search" size={17} color="#07110F" /></Pressable>
      </View>
      {recall.slice(0, 5).map(renderObservation)}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: '#081A14', borderColor: '#345C35', borderWidth: 1, borderRadius: 18, padding: 16, marginTop: 14, gap: 9 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  kicker: { color: '#B7FF56', fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  title: { color: '#F1F8F3', fontSize: 16, fontWeight: '800', marginTop: 4 },
  refresh: { padding: 10, borderRadius: 10, backgroundColor: '#122A20' },
  metrics: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  metric: { color: '#C3D6CD', fontSize: 10, backgroundColor: '#12261F', paddingHorizontal: 8, paddingVertical: 5, borderRadius: 8 },
  deviceNote: { color: '#789187', fontSize: 9 },
  changeText: { color: '#DDE9E4', fontSize: 12, lineHeight: 17, backgroundColor: '#10271C', padding: 10, borderRadius: 10 },
  empty: { color: '#8BA397', fontSize: 12 },
  observation: { flexDirection: 'row', alignItems: 'center', gap: 8, borderTopWidth: 1, borderColor: '#1B352B', paddingTop: 8 },
  observationBody: { flex: 1, gap: 2 },
  observationTitle: { color: '#F3F8F5', fontSize: 13, fontWeight: '800', textTransform: 'capitalize' },
  observationMeta: { color: '#9BB3A8', fontSize: 10 },
  observationDescription: { color: '#C1D1C9', fontSize: 11 },
  removeButton: { padding: 8 },
  divider: { height: 1, backgroundColor: '#274338', marginVertical: 5 },
  sectionTitle: { color: '#EAF3EE', fontSize: 13, fontWeight: '800' },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  chip: { backgroundColor: '#142A22', borderColor: '#294539', borderWidth: 1, paddingHorizontal: 9, paddingVertical: 6, borderRadius: 9 },
  chipActive: { backgroundColor: '#B7FF56', borderColor: '#B7FF56' },
  chipText: { color: '#C5D6CE', fontSize: 10, fontWeight: '700', textTransform: 'capitalize' },
  chipTextActive: { color: '#07110F' },
  input: { color: '#F3F8F5', backgroundColor: '#10231C', borderWidth: 1, borderColor: '#294539', borderRadius: 10, paddingHorizontal: 10, paddingVertical: 9, fontSize: 12 },
  verifyRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  verifyText: { color: '#C6D7CE', fontSize: 11, flex: 1 },
  submit: { backgroundColor: '#B7FF56', paddingVertical: 10, borderRadius: 10, alignItems: 'center' },
  disabled: { opacity: 0.45 },
  submitText: { color: '#07110F', fontWeight: '900', fontSize: 12 },
  notice: { color: '#B7FF56', fontSize: 11, lineHeight: 15 },
  searchRow: { flexDirection: 'row', gap: 7 },
  searchInput: { flex: 1 },
  searchButton: { backgroundColor: '#B7FF56', paddingHorizontal: 13, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
});

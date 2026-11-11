import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useMemo, useRef, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import HandTracker, { type SignContextState } from '../components/HandTracker';
import { SignRecorder, downloadTrainingSample } from '../src/landmarkRecorder.web';

/**
 * Collect labelled sign samples for training and evaluation.
 *
 * Saves landmark numbers only, never video, as one JSON file per sample on
 * this computer. Labels are typed by a person who knows the sign; nothing is
 * guessed. Web only: sign tracking does not run in the native apps.
 */
export default function RecordSignsScreen() {
  const recorder = useMemo(() => new SignRecorder(20), []);
  const [signerId, setSignerId] = useState('');
  const [glosses, setGlosses] = useState('');
  const [english, setEnglish] = useState('');
  const [consent, setConsent] = useState(false);
  const [verified, setVerified] = useState(false);
  const [recording, setRecording] = useState(false);
  const [frames, setFrames] = useState(0);
  const [hands, setHands] = useState(0);
  const [notice, setNotice] = useState('');
  const [saved, setSaved] = useState(0);
  const startedAt = useRef(0);

  const onContext = (state: SignContextState) => {
    setHands(state.handCount);
    if (recorder.recording) setFrames(recorder.frameCount);
  };

  const start = () => {
    if (!signerId.trim()) return setNotice('Enter a signer ID first (a code, not a name).');
    if (!consent) return setNotice('The signer must agree to the recording first.');
    recorder.start();
    startedAt.current = Date.now();
    setFrames(0);
    setRecording(true);
    setNotice('Recording. Sign now, then press Stop.');
  };

  const discard = () => {
    recorder.discard();
    setRecording(false);
    setFrames(0);
    setNotice('Discarded.');
  };

  const stop = () => {
    const labels = glosses.split(/[\s,]+/).map((value) => value.trim()).filter(Boolean);
    if (!labels.length) return setNotice('Type the gloss labels for what was signed, then press Stop again.');
    if (!verified) return setNotice('Confirm that a person who knows the sign checked the labels.');
    try {
      const sample = recorder.stop({
        id: `${signerId.trim()}-${labels.join('_').toLowerCase()}-${new Date().toISOString().replace(/[:.]/g, '-')}`,
        signerId: signerId.trim(),
        glosses: labels,
        text: english,
      });
      downloadTrainingSample(sample);
      setSaved((count) => count + 1);
      setNotice(`Saved ${sample.frames.length} frames (${((Date.now() - startedAt.current) / 1000).toFixed(1)} s) for ${sample.glosses.join(' ')}.`);
      setVerified(false);
    } catch (error) {
      setNotice((error as Error).message);
    }
    setRecording(false);
    setFrames(0);
  };

  if (Platform.OS !== 'web') {
    return (
      <SafeAreaView style={s.safe}>
        <Text style={s.body}>Sign recording runs in the web app, where hand tracking is available.</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={s.safe}>
      <ScrollView contentContainerStyle={s.page} keyboardShouldPersistTaps="handled">
        <View style={s.header}>
          <Pressable onPress={() => router.back()} style={s.back} accessibilityRole="button" accessibilityLabel="Back">
            <Ionicons name="arrow-back" size={20} color="#FFFFFF" />
          </Pressable>
          <View style={s.headerCopy}>
            <Text style={s.kicker}>SIGN DATA · FOR THE TEAM</Text>
            <Text style={s.title}>Record labelled signs</Text>
          </View>
          <Text style={s.count}>{saved} saved</Text>
        </View>
        <Text style={s.body}>
          Saves hand, body and face landmark numbers as a JSON file on this computer. No video is recorded or uploaded.
          Use a code for the signer, not a name, and keep one signer's files out of both training and testing.
        </Text>

        <View style={s.camera}>
          <HandTracker onContextChange={onContext} trainingRecorder={recorder} showLandmarks />
          <View style={[s.badge, recording && s.badgeLive]}>
            <Text style={s.badgeText}>{recording ? `● REC · ${frames} frames` : hands ? `${hands} hand${hands === 1 ? '' : 's'} tracked` : 'Show your hands'}</Text>
          </View>
        </View>

        <TextInput style={s.input} value={signerId} onChangeText={setSignerId} placeholder="Signer ID (for example S01)" placeholderTextColor="#789187" autoCapitalize="characters" accessibilityLabel="Signer ID" />
        <TextInput style={s.input} value={glosses} onChangeText={setGlosses} placeholder="Gloss labels in order (for example ME NEED WATER)" placeholderTextColor="#789187" autoCapitalize="characters" accessibilityLabel="Gloss labels" />
        <TextInput style={s.input} value={english} onChangeText={setEnglish} placeholder="English meaning (optional)" placeholderTextColor="#789187" accessibilityLabel="English meaning" />

        <Pressable style={s.check} onPress={() => setConsent((value) => !value)} accessibilityRole="checkbox" accessibilityState={{ checked: consent }}>
          <Ionicons name={consent ? 'checkbox' : 'square-outline'} size={20} color="#B7FF56" />
          <Text style={s.checkText}>The signer agreed to this recording being used to train and test GuideLens.</Text>
        </Pressable>
        <Pressable style={s.check} onPress={() => setVerified((value) => !value)} accessibilityRole="checkbox" accessibilityState={{ checked: verified }}>
          <Ionicons name={verified ? 'checkbox' : 'square-outline'} size={20} color="#B7FF56" />
          <Text style={s.checkText}>A person who knows ASL checked that the labels match what was signed.</Text>
        </Pressable>

        <View style={s.actions}>
          {!recording ? (
            <Pressable style={s.primary} onPress={start} accessibilityRole="button" accessibilityLabel="Start recording">
              <Ionicons name="radio-button-on" size={18} color="#07110F" />
              <Text style={s.primaryText}>Start recording</Text>
            </Pressable>
          ) : (
            <>
              <Pressable style={s.primary} onPress={stop} accessibilityRole="button" accessibilityLabel="Stop and save">
                <Ionicons name="download-outline" size={18} color="#07110F" />
                <Text style={s.primaryText}>Stop and save</Text>
              </Pressable>
              <Pressable style={s.secondary} onPress={discard} accessibilityRole="button" accessibilityLabel="Discard this recording">
                <Text style={s.secondaryText}>Discard</Text>
              </Pressable>
            </>
          )}
        </View>
        {notice ? <Text style={s.notice} accessibilityLiveRegion="polite">{notice}</Text> : null}
        <Text style={s.hint}>
          Train with: python apps/api/training/train_sign_transformer.py --train samples.jsonl (one saved file per line).
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#07110F' },
  page: { padding: 16, gap: 12, maxWidth: 720, width: '100%', alignSelf: 'center' },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  back: { width: 38, height: 38, borderRadius: 19, backgroundColor: '#11221D', alignItems: 'center', justifyContent: 'center' },
  headerCopy: { flex: 1 },
  kicker: { color: '#B7FF56', fontSize: 10, fontWeight: '900', letterSpacing: 1.2 },
  title: { color: '#FFFFFF', fontSize: 20, fontWeight: '800' },
  count: { color: '#D5E6D8', fontSize: 12, fontWeight: '800' },
  body: { color: '#9EB5AA', fontSize: 13, lineHeight: 19 },
  camera: { height: 320, borderRadius: 14, overflow: 'hidden', backgroundColor: '#050D0B' },
  badge: { position: 'absolute', left: 10, top: 10, backgroundColor: 'rgba(7,17,15,.85)', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  badgeLive: { backgroundColor: '#C8352B' },
  badgeText: { color: '#FFFFFF', fontSize: 12, fontWeight: '800' },
  input: { height: 44, backgroundColor: '#11221D', borderRadius: 10, paddingHorizontal: 12, color: '#FFFFFF', fontSize: 14, borderWidth: 1, borderColor: '#244537' },
  check: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  checkText: { color: '#D5E6D8', fontSize: 13, flex: 1 },
  actions: { flexDirection: 'row', gap: 10 },
  primary: { flex: 1, flexDirection: 'row', gap: 8, alignItems: 'center', justifyContent: 'center', backgroundColor: '#B7FF56', borderRadius: 12, paddingVertical: 13 },
  primaryText: { color: '#07110F', fontSize: 15, fontWeight: '900' },
  secondary: { paddingHorizontal: 18, justifyContent: 'center', borderWidth: 1, borderColor: '#2B4237', borderRadius: 12 },
  secondaryText: { color: '#D5E6D8', fontWeight: '800' },
  notice: { color: '#FFD043', fontSize: 13, fontWeight: '700' },
  hint: { color: '#607B6E', fontSize: 11 },
});

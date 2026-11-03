import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AssistMode, useVisionLoop } from '../src/vision';

type DetectionBox = {
  l: `${number}%`;
  t: `${number}%`;
  w: `${number}%`;
  h: `${number}%`;
  label: string;
};

type AssistFrame = {
  title: string;
  detail: string;
  speech: string;
  icon: keyof typeof Ionicons.glyphMap;
  boxes: DetectionBox[];
};

const sceneFrames: AssistFrame[] = [
  {
    title: 'Clear path ahead',
    detail: 'Door 13 feet ahead. Chair to your right.',
    speech: 'Door 13 feet ahead. Chair to your right.',
    icon: 'navigate',
    boxes: [
      { l: '12%', t: '25%', w: '22%', h: '32%', label: 'chair · 93%' },
      { l: '61%', t: '18%', w: '25%', h: '49%', label: 'door · 97%' },
    ],
  },
  {
    title: 'Person approaching',
    detail: 'A person is about 10 feet ahead.',
    speech: 'A person is about 10 feet ahead.',
    icon: 'person',
    boxes: [{ l: '38%', t: '19%', w: '26%', h: '55%', label: 'person · 96%' }],
  },
  {
    title: 'Table ahead',
    detail: 'A table is 5 feet ahead. A cup is on top.',
    speech: 'A table is 5 feet ahead. A cup is on top.',
    icon: 'cafe',
    boxes: [
      { l: '15%', t: '44%', w: '70%', h: '31%', label: 'table · 95%' },
      { l: '57%', t: '35%', w: '12%', h: '12%', label: 'cup · 89%' },
    ],
  },
];

const signFrames: AssistFrame[] = [
  {
    title: 'Hello, nice to meet you.',
    detail: 'Text interpretation · Sign language demo',
    speech: 'Hello, nice to meet you.',
    icon: 'hand-left',
    boxes: [
      { l: '35%', t: '13%', w: '29%', h: '57%', label: 'signer · framed' },
      { l: '23%', t: '43%', w: '18%', h: '22%', label: 'left hand' },
      { l: '61%', t: '43%', w: '17%', h: '22%', label: 'right hand' },
    ],
  },
  {
    title: 'Please speak more slowly.',
    detail: 'Text interpretation · Sign language demo',
    speech: 'Please speak more slowly.',
    icon: 'chatbox-ellipses',
    boxes: [
      { l: '36%', t: '14%', w: '28%', h: '56%', label: 'signer · framed' },
      { l: '25%', t: '39%', w: '18%', h: '23%', label: 'left hand' },
      { l: '58%', t: '35%', w: '18%', h: '23%', label: 'right hand' },
    ],
  },
  {
    title: 'Thank you for your help.',
    detail: 'Text interpretation · Sign language demo',
    speech: 'Thank you for your help.',
    icon: 'heart',
    boxes: [
      { l: '37%', t: '13%', w: '28%', h: '57%', label: 'signer · framed' },
      { l: '26%', t: '42%', w: '17%', h: '23%', label: 'left hand' },
      { l: '59%', t: '42%', w: '17%', h: '23%', label: 'right hand' },
    ],
  },
];

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<AssistMode>('scene');
  const [frameIndex, setFrameIndex] = useState(0);
  const [speaking, setSpeaking] = useState(true);
  const [torchEnabled, setTorchEnabled] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const { result: vision, state: visionState, scanNow } = useVisionLoop(cameraRef, mode, Boolean(permission?.granted));

  const frames = mode === 'scene' ? sceneFrames : signFrames;
  const fallback = frames[frameIndex % frames.length];
  const current: AssistFrame = vision ? {
    title: vision.summary,
    detail: vision.speech,
    speech: vision.speech,
    icon: mode === 'sign' ? 'hand-left' : vision.urgency === 'high' ? 'warning' : 'navigate',
    boxes: [],
  } : fallback;
  const isSignMode = mode === 'sign';

  const speak = (text = current.speech) => {
    if (!speaking) return;
    Speech.stop();
    Speech.speak(text, { language: 'en-US', rate: 0.92 });
  };

  const switchMode = (nextMode: AssistMode) => {
    if (nextMode === mode) return;
    Speech.stop();
    setMode(nextMode);
    setFrameIndex(0);
  };

  useEffect(() => {
    if (vision?.changed || vision?.urgency === 'high') speak(vision.speech);
  }, [vision?.scene_key]);

  useEffect(() => () => { void Speech.stop(); }, []);

  if (!permission) return <View style={c.fallback} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={c.permission}>
        <View style={c.permissionIcon}><Ionicons name="camera" size={34} color="#B7FF56" /></View>
        <Text style={c.permissionTitle}>Let your camera help communicate</Text>
        <Text style={c.permissionText}>GuideLens needs camera access for scene assistance and the sign language demo. This prototype does not save video.</Text>
        <Pressable onPress={requestPermission} style={c.allow}><Text style={c.allowText}>Allow camera access</Text></Pressable>
        <Pressable onPress={() => router.back()}><Text style={c.backText}>Go back</Text></Pressable>
      </SafeAreaView>
    );
  }

  return (
    <View style={c.root}>
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" enableTorch={torchEnabled} />
      <View style={c.shade} />
      <SafeAreaView style={c.overlay}>
        <View style={c.top}>
          <Pressable accessibilityLabel="Go back" onPress={() => router.back()} style={c.round}>
            <Ionicons name="chevron-back" size={24} color="#fff" />
          </Pressable>
          <View style={c.modeIndicator}>
            <View style={c.greenDot} />
            <Text style={c.modeText}>{visionState === 'offline' ? 'VISION · RECONNECTING' : visionState === 'analyzing' ? 'VISION · ANALYZING' : isSignMode ? 'SIGN INTERPRETATION · LIVE' : 'GROQ VISION · LIVE'}</Text>
          </View>
          <Pressable accessibilityLabel="Toggle spoken output" onPress={() => { setSpeaking((value) => !value); Speech.stop(); }} style={c.round}>
            <Ionicons name={speaking ? 'volume-high' : 'volume-mute'} size={21} color="#fff" />
          </Pressable>
        </View>

        <View style={c.modeTabs} accessibilityRole="tablist">
          <Pressable accessibilityRole="tab" accessibilityState={{ selected: mode === 'scene' }} onPress={() => switchMode('scene')} style={[c.modeTab, mode === 'scene' && c.modeTabActive]}>
            <Ionicons name="eye-outline" size={17} color={mode === 'scene' ? '#09130D' : '#DDE9E4'} />
            <Text style={[c.modeTabText, mode === 'scene' && c.modeTabTextActive]}>Scene assist</Text>
          </Pressable>
          <Pressable accessibilityRole="tab" accessibilityState={{ selected: mode === 'sign' }} onPress={() => switchMode('sign')} style={[c.modeTab, mode === 'sign' && c.modeTabActive]}>
            <Ionicons name="hand-left-outline" size={17} color={mode === 'sign' ? '#09130D' : '#DDE9E4'} />
            <Text style={[c.modeTabText, mode === 'sign' && c.modeTabTextActive]}>Sign to voice</Text>
          </Pressable>
        </View>

        {isSignMode && (
          <View style={c.signGuide}>
            <Ionicons name="scan-outline" size={16} color="#C8FF72" />
            <Text style={c.signGuideText}>Keep hands, face, and upper body in frame.</Text>
          </View>
        )}

        <View style={c.reticle}>
          {current.boxes.map((box, index) => (
            <View key={`${mode}-${index}`} style={[c.box, { left: box.l, top: box.t, width: box.w, height: box.h }, isSignMode && c.signBox]}>
              <Text style={c.boxLabel}>{box.label}</Text>
            </View>
          ))}
        </View>

        <View style={c.bottom}>
          <View style={[c.answer, isSignMode && c.signAnswer]}>
            <View style={[c.answerIcon, isSignMode && c.signAnswerIcon]}><Ionicons name={current.icon} size={25} color="#07110F" /></View>
            <View style={c.answerCopy}>
              <Text style={isSignMode ? c.translationLabel : c.answerTitle}>{isSignMode ? 'SIGN TO TEXT · DEMO' : current.title}</Text>
              <Text style={isSignMode ? c.translationText : c.answerText}>{isSignMode ? current.title : current.detail}</Text>
              {isSignMode && <Text style={c.translationDetail}>{current.detail}</Text>}
            </View>
          </View>

          {isSignMode && (
            <View style={c.confirmation}>
              <Ionicons name="information-circle-outline" size={14} color="#D3E0DA" />
              <Text style={c.confirmationText}>Confirm the meaning with the signer before relying on this interpretation.</Text>
            </View>
          )}

          <View style={c.controls}>
            <Pressable onPress={scanNow} style={c.control}>
              <Ionicons name="refresh" size={22} color="#EAF6F0" />
              <Text style={c.controlText}>{isSignMode ? 'Read again' : 'Scan again'}</Text>
            </Pressable>
            <Pressable accessibilityLabel={isSignMode ? 'Speak translation' : 'Repeat description'} onPress={() => speak()} style={c.speak}>
              <Ionicons name={isSignMode ? 'megaphone' : 'volume-high'} size={24} color="#07110F" />
            </Pressable>
            <Pressable accessibilityLabel={torchEnabled ? 'Turn flashlight off' : 'Turn flashlight on'} onPress={() => setTorchEnabled((value) => !value)} style={c.control}>
              <Ionicons name={torchEnabled ? 'flash' : 'flash-off'} size={22} color={torchEnabled ? '#C8FF72' : '#EAF6F0'} />
              <Text style={[c.controlText, torchEnabled && c.controlTextActive]}>{torchEnabled ? 'Flash on' : 'Flashlight'}</Text>
            </Pressable>
          </View>
          <Text style={c.hint}>{isSignMode ? 'Tap the megaphone to speak the interpreted text aloud' : 'Tap the speaker button to repeat the description'}</Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

const c = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#07110F' },
  fallback: { flex: 1, backgroundColor: '#07110F' },
  shade: { position: 'absolute', top: 0, right: 0, bottom: 0, left: 0, backgroundColor: 'rgba(2,10,8,.18)' },
  overlay: { flex: 1, paddingHorizontal: 18 },
  top: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  round: { width: 46, height: 46, borderRadius: 23, backgroundColor: 'rgba(4,13,10,.62)', alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,.16)' },
  modeIndicator: { backgroundColor: 'rgba(4,13,10,.76)', borderRadius: 99, paddingHorizontal: 13, paddingVertical: 9, flexDirection: 'row', alignItems: 'center', gap: 7 },
  greenDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#B7FF56' },
  modeText: { color: '#EAF5EF', fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  modeTabs: { flexDirection: 'row', alignSelf: 'center', gap: 6, backgroundColor: 'rgba(4,13,10,.74)', borderRadius: 17, borderWidth: 1, borderColor: 'rgba(255,255,255,.12)', padding: 5, marginTop: 16 },
  modeTab: { minHeight: 42, flexDirection: 'row', alignItems: 'center', gap: 7, paddingHorizontal: 13, borderRadius: 12 },
  modeTabActive: { backgroundColor: '#B7FF56' },
  modeTabText: { color: '#DDE9E4', fontSize: 12, fontWeight: '800' },
  modeTabTextActive: { color: '#09130D' },
  signGuide: { alignSelf: 'center', flexDirection: 'row', alignItems: 'center', gap: 7, borderRadius: 99, paddingHorizontal: 12, paddingVertical: 8, marginTop: 11, backgroundColor: 'rgba(5,20,14,.8)', borderWidth: 1, borderColor: 'rgba(183,255,86,.35)' },
  signGuideText: { color: '#E5F0E9', fontSize: 11, fontWeight: '700' },
  reticle: { flex: 1, marginVertical: 12, position: 'relative' },
  box: { position: 'absolute', borderWidth: 2, borderColor: '#B7FF56', borderRadius: 10 },
  signBox: { borderColor: '#C8FF72' },
  boxLabel: { position: 'absolute', top: -24, left: -2, backgroundColor: '#B7FF56', color: '#07110F', fontWeight: '800', fontSize: 10, paddingHorizontal: 7, paddingVertical: 4, borderRadius: 5 },
  bottom: { paddingBottom: 10 },
  answer: { flexDirection: 'row', alignItems: 'center', gap: 13, backgroundColor: 'rgba(5,18,14,.9)', borderWidth: 1, borderColor: 'rgba(183,255,86,.28)', borderRadius: 20, padding: 16 },
  signAnswer: { borderColor: 'rgba(200,255,114,.62)', backgroundColor: 'rgba(5,26,17,.94)' },
  answerIcon: { width: 48, height: 48, borderRadius: 15, backgroundColor: '#B7FF56', alignItems: 'center', justifyContent: 'center' },
  signAnswerIcon: { backgroundColor: '#C8FF72' },
  answerCopy: { flex: 1 },
  answerTitle: { color: '#F7FBF8', fontSize: 18, fontWeight: '800' },
  answerText: { color: '#AFC1BA', fontSize: 13, lineHeight: 19, marginTop: 4 },
  translationLabel: { color: '#C8FF72', fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  translationText: { color: '#F7FBF8', fontSize: 20, lineHeight: 26, fontWeight: '800', marginTop: 5 },
  translationDetail: { color: '#AFC1BA', fontSize: 11, lineHeight: 16, marginTop: 5 },
  confirmation: { flexDirection: 'row', alignItems: 'flex-start', gap: 6, paddingHorizontal: 8, marginTop: 9 },
  confirmationText: { flex: 1, color: '#D3E0DA', fontSize: 10, lineHeight: 14 },
  controls: { flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center', marginTop: 17 },
  control: { alignItems: 'center', width: 90, gap: 5 },
  controlText: { color: '#DDE9E4', fontSize: 10, fontWeight: '700' },
  controlTextActive: { color: '#C8FF72' },
  speak: { width: 66, height: 66, borderRadius: 33, backgroundColor: '#B7FF56', alignItems: 'center', justifyContent: 'center', borderWidth: 5, borderColor: 'rgba(183,255,86,.25)' },
  hint: { color: 'rgba(255,255,255,.62)', textAlign: 'center', fontSize: 10, marginTop: 12 },
  permission: { flex: 1, backgroundColor: '#07110F', alignItems: 'center', justifyContent: 'center', padding: 30 },
  permissionIcon: { width: 76, height: 76, borderRadius: 24, backgroundColor: '#142D24', alignItems: 'center', justifyContent: 'center' },
  permissionTitle: { color: '#F4FAF6', fontSize: 26, fontWeight: '800', marginTop: 24, textAlign: 'center' },
  permissionText: { color: '#94A9A0', fontSize: 15, lineHeight: 23, textAlign: 'center', maxWidth: 430, marginTop: 12 },
  allow: { backgroundColor: '#B7FF56', borderRadius: 14, paddingHorizontal: 24, paddingVertical: 15, marginTop: 28 },
  allowText: { color: '#07110F', fontWeight: '800' },
  backText: { color: '#91A79E', marginTop: 18 },
});

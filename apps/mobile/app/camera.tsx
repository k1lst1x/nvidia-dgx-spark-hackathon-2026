import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import HandTracker, { type SignContextState } from '../components/HandTracker';
import { AssistMode, useVisionLoop } from '../src/vision';

const emptySignContext: SignContextState = { frameCount: 0, maxFrames: 90, handCount: 0, ready: false };

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

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<AssistMode>('scene');
  const [frameIndex, setFrameIndex] = useState(0);
  const [speaking, setSpeaking] = useState(true);
  const [torchEnabled, setTorchEnabled] = useState(false);
  const [signContext, setSignContext] = useState<SignContextState>(emptySignContext);
  const [isAnalysingContext, setIsAnalysingContext] = useState(false);
  const [contextMessage, setContextMessage] = useState('');
  const [captureVersion, setCaptureVersion] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const analysisTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const capturedSequence = useRef<number[][]>([]);
  const cameraRef = useRef<CameraView>(null);

  const isSignMode = mode === 'sign';
  const usesBrowserHandTracking = isSignMode && Platform.OS === 'web';
  const { result: vision, state: visionState, detections, scanNow } = useVisionLoop(
    cameraRef,
    mode,
    Boolean(permission?.granted) && !usesBrowserHandTracking,
  );
  const frames = sceneFrames;
  const fallback = frames[frameIndex % frames.length];
  const current: AssistFrame = vision ? {
    title: vision.summary,
    detail: vision.speech,
    speech: vision.speech,
    icon: mode === 'sign' ? 'hand-left' : vision.urgency === 'high' ? 'warning' : 'navigate',
    boxes: detections.map((item) => ({
      l: `${Math.round(item.x * 100)}%` as `${number}%`,
      t: `${Math.round(item.y * 100)}%` as `${number}%`,
      w: `${Math.round(item.width * 100)}%` as `${number}%`,
      h: `${Math.round(item.height * 100)}%` as `${number}%`,
      label: `${item.label} · ${Math.round(item.confidence * 100)}%`,
    })),
  } : fallback;
  const browserSignHeading = signContext.error
    ? 'Hand tracking needs attention.'
    : isAnalysingContext
      ? 'Reviewing the full signing sequence…'
      : contextMessage || (signContext.ready ? 'Context window ready to analyze.' : 'Building context from continuous signing…');
  const browserSignDetail = signContext.error
    ? signContext.error
    : signContext.ready
      ? `${signContext.handCount} hands tracked across ${signContext.maxFrames} frames. Analyze the whole sequence, not one gesture.`
      : `${signContext.handCount} hands tracked · ${signContext.frameCount}/${signContext.maxFrames} contextual frames`;
  const signHeading = usesBrowserHandTracking
    ? browserSignHeading
    : vision?.summary || (visionState === 'offline' ? 'Sign interpretation is reconnecting…' : 'Looking for a signed message…');
  const signDetail = usesBrowserHandTracking
    ? browserSignDetail
    : vision?.speech || 'Keep both hands clearly in the camera frame.';

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
    setSignContext(emptySignContext);
    capturedSequence.current = [];
    setContextMessage('');
    setIsAnalysingContext(false);
    setCaptureVersion((value) => value + 1);
  };

  const restartContextCapture = () => {
    if (!usesBrowserHandTracking) {
      scanNow();
      return;
    }
    if (analysisTimer.current) clearTimeout(analysisTimer.current);
    setSignContext(emptySignContext);
    capturedSequence.current = [];
    setContextMessage('');
    setIsAnalysingContext(false);
    setCaptureVersion((value) => value + 1);
  };

  const analyzeSignContext = () => {
    if (!signContext.ready || capturedSequence.current.length !== signContext.maxFrames || isAnalysingContext) return;
    setIsAnalysingContext(true);
    setContextMessage('');
    analysisTimer.current = setTimeout(() => {
      setIsAnalysingContext(false);
      setContextMessage('Context captured. Ready for a temporal sign-language model to interpret the full signed thought.');
    }, 900);
  };

  useEffect(() => {
    if (mode === 'scene') timer.current = setInterval(() => setFrameIndex((value) => (value + 1) % frames.length), 5500);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [mode, frames.length]);

  useEffect(() => {
    if (!usesBrowserHandTracking && vision && (vision.changed || vision.urgency === 'high')) speak(vision.speech);
  }, [usesBrowserHandTracking, vision?.scene_key]);

  useEffect(() => () => {
    if (analysisTimer.current) clearTimeout(analysisTimer.current);
    void Speech.stop();
  }, []);

  if (!permission) return <View style={c.fallback} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={c.permission}>
        <View style={c.permissionIcon}><Ionicons name="camera" size={34} color="#B7FF56" /></View>
        <Text style={c.permissionTitle}>Let your camera help communicate</Text>
        <Text style={c.permissionText}>GuideLens needs camera access for scene assistance and sign-context capture. This prototype does not save video.</Text>
        <Pressable onPress={requestPermission} style={c.allow}><Text style={c.allowText}>Allow camera access</Text></Pressable>
        <Pressable onPress={() => router.back()}><Text style={c.backText}>Go back</Text></Pressable>
      </SafeAreaView>
    );
  }

  return (
    <View style={c.root}>
      {usesBrowserHandTracking ? <HandTracker key={captureVersion} onContextChange={setSignContext} onContextCaptured={(sequence) => { capturedSequence.current = sequence; }} /> : <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" enableTorch={torchEnabled} />}
      <View style={c.shade} />
      <SafeAreaView style={c.overlay}>
        <View style={c.top}>
          <Pressable accessibilityLabel="Go back" onPress={() => router.back()} style={c.round}>
            <Ionicons name="chevron-back" size={24} color="#fff" />
          </Pressable>
          <View style={c.modeIndicator}>
            <View style={c.greenDot} />
            <Text style={c.modeText}>{usesBrowserHandTracking ? 'SIGN CONTEXT' : visionState === 'offline' ? 'VISION · RECONNECTING' : visionState === 'analyzing' ? 'VISION · ANALYZING' : isSignMode ? 'SIGN INTERPRETATION · LIVE' : 'GROQ VISION · LIVE'}</Text>
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

        <View style={c.reticle}>
          {!usesBrowserHandTracking && current.boxes.map((box, index) => (
            <View key={`${mode}-${index}`} style={[c.box, { left: box.l, top: box.t, width: box.w, height: box.h }, isSignMode && c.signBox]}>
              <Text style={c.boxLabel}>{box.label}</Text>
            </View>
          ))}
        </View>

        <View style={c.bottom}>
          <View style={[c.answer, isSignMode && c.signAnswer]}>
            <View style={[c.answerIcon, isSignMode && c.signAnswerIcon]}><Ionicons name={isSignMode ? 'hand-left' : current.icon} size={25} color="#07110F" /></View>
            <View style={c.answerCopy}>
              <Text style={isSignMode ? c.translationText : c.answerTitle}>{isSignMode ? signHeading : current.title}</Text>
              <Text style={isSignMode ? c.translationDetail : c.answerText}>{isSignMode ? signDetail : current.detail}</Text>
            </View>
          </View>

          <View style={c.controls}>
            <Pressable onPress={restartContextCapture} style={c.control}>
              <Ionicons name="refresh" size={22} color="#EAF6F0" />
              <Text style={c.controlText}>{usesBrowserHandTracking ? 'New context' : isSignMode ? 'Read again' : 'Scan again'}</Text>
            </Pressable>
            <Pressable accessibilityLabel={usesBrowserHandTracking ? 'Analyze sign language context' : isSignMode ? 'Speak translation' : 'Repeat description'} onPress={usesBrowserHandTracking ? analyzeSignContext : () => speak()} disabled={usesBrowserHandTracking && (!signContext.ready || isAnalysingContext)} style={[c.speak, usesBrowserHandTracking && (!signContext.ready || isAnalysingContext) && c.speakDisabled]}>
              <Ionicons name={usesBrowserHandTracking ? 'sparkles' : isSignMode ? 'megaphone' : 'volume-high'} size={24} color="#07110F" />
            </Pressable>
            <Pressable accessibilityLabel={torchEnabled ? 'Turn flashlight off' : 'Turn flashlight on'} onPress={() => setTorchEnabled((value) => !value)} style={c.control}>
              <Ionicons name={torchEnabled ? 'flash' : 'flash-off'} size={22} color={torchEnabled ? '#C8FF72' : '#EAF6F0'} />
              <Text style={[c.controlText, torchEnabled && c.controlTextActive]}>{torchEnabled ? 'Flash on' : 'Flashlight'}</Text>
            </Pressable>
          </View>
          <Text style={c.hint}>{usesBrowserHandTracking ? (signContext.ready ? 'Tap the center button to analyze the full context window' : 'Build a complete context window before analysis') : isSignMode ? 'Tap the megaphone to speak the interpreted text aloud' : 'Tap the speaker button to repeat the description'}</Text>
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
  translationText: { color: '#F7FBF8', fontSize: 20, lineHeight: 26, fontWeight: '800', marginTop: 5 },
  translationDetail: { color: '#AFC1BA', fontSize: 11, lineHeight: 16, marginTop: 5 },
  controls: { flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center', marginTop: 17 },
  control: { alignItems: 'center', width: 90, gap: 5 },
  controlText: { color: '#DDE9E4', fontSize: 10, fontWeight: '700' },
  controlTextActive: { color: '#C8FF72' },
  speak: { width: 66, height: 66, borderRadius: 33, backgroundColor: '#B7FF56', alignItems: 'center', justifyContent: 'center', borderWidth: 5, borderColor: 'rgba(183,255,86,.25)' },
  speakDisabled: { opacity: 0.42 },
  hint: { color: 'rgba(255,255,255,.62)', textAlign: 'center', fontSize: 10, marginTop: 12 },
  permission: { flex: 1, backgroundColor: '#07110F', alignItems: 'center', justifyContent: 'center', padding: 30 },
  permissionIcon: { width: 76, height: 76, borderRadius: 24, backgroundColor: '#142D24', alignItems: 'center', justifyContent: 'center' },
  permissionTitle: { color: '#F4FAF6', fontSize: 26, fontWeight: '800', marginTop: 24, textAlign: 'center' },
  permissionText: { color: '#94A9A0', fontSize: 15, lineHeight: 23, textAlign: 'center', maxWidth: 430, marginTop: 12 },
  allow: { backgroundColor: '#B7FF56', borderRadius: 14, paddingHorizontal: 24, paddingVertical: 15, marginTop: 28 },
  allowText: { color: '#07110F', fontWeight: '800' },
  backText: { color: '#91A79E', marginTop: 18 },
});

import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import HandTracker, { type SignContextState } from '../components/HandTracker';
import { offlinePhraseCount, resolveOfflinePhrase } from '../src/offlinePhrasebook';
import { AssistMode, hasVisionBackend, interpretSignSequence, useVisionLoop } from '../src/vision';

const emptySignContext: SignContextState = { frameCount: 0, maxFrames: 180, handCount: 0, ready: false };

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

function formatAslWord(word: string): string {
  if (!word) return '';
  const special: Record<string, string> = {
    thankyou: 'Thank you',
    hello: 'Hello',
    please: 'Please',
    yes: 'Yes',
    no: 'No',
    water: 'Water',
    alligator: 'Alligator',
    airplane: 'Airplane',
    fireman: 'Firefighter',
  };
  return special[word.toLowerCase()] || word.charAt(0).toUpperCase() + word.slice(1);
}

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<AssistMode>('scene');
  const [speaking, setSpeaking] = useState(true);
  const [torchEnabled, setTorchEnabled] = useState(false);
  const [signContext, setSignContext] = useState<SignContextState>(emptySignContext);
  const [isAnalysingContext, setIsAnalysingContext] = useState(false);
  const [translation, setTranslation] = useState('');
  const [translationError, setTranslationError] = useState('');
  const [captureVersion, setCaptureVersion] = useState(0);
  const translationInFlight = useRef(false);
  const lastTranslation = useRef('');
  const lastOfflinePhrase = useRef('');
  const lastOfflineWordPhrase = useRef('');
  const capturedSequence = useRef<number[][]>([]);
  const capturedFrames = useRef<string[]>([]);
  const cameraRef = useRef<CameraView>(null);
  const aslEngineRef = useRef<{
    backspace: () => string;
    addSpace: () => string;
    clearText: () => void;
  } | null>(null);

  const isSignMode = mode === 'sign';
  const usesBrowserHandTracking = isSignMode && Platform.OS === 'web';
  const { result: vision, state: visionState, detections, scanNow } = useVisionLoop(
    cameraRef,
    mode,
    Boolean(permission?.granted) && !usesBrowserHandTracking,
  );
  const fallback: AssistFrame = {
    title: visionState === 'offline' ? 'Local vision is reconnecting…' : 'Looking for nearby objects…',
    detail: 'Point the camera at an object and keep it steady for a moment.',
    speech: 'No objects detected yet.',
    icon: visionState === 'offline' ? 'cloud-offline-outline' : 'scan-outline',
    boxes: [],
  };
  const objectCounts = detections.reduce<Record<string, number>>((counts, item) => {
    counts[item.label] = (counts[item.label] || 0) + 1;
    return counts;
  }, {});
  const localSummary = detections.length
    ? `Detected ${Object.entries(objectCounts).slice(0, 3).map(([label, count]) => count > 1 ? `${count} ${label}s` : label).join(', ')}`
    : 'Looking for nearby objects…';
  const current: AssistFrame = vision || detections.length ? {
    title: vision?.summary || localSummary,
    detail: vision?.speech || 'Object detection is running privately on this device.',
    speech: vision?.speech || localSummary,
    icon: mode === 'sign' ? 'hand-left' : vision?.urgency === 'high' ? 'warning' : 'navigate',
    boxes: detections.map((item) => ({
      l: `${Math.round(item.x * 100)}%` as `${number}%`,
      t: `${Math.round(item.y * 100)}%` as `${number}%`,
      w: `${Math.round(item.width * 100)}%` as `${number}%`,
      h: `${Math.round(item.height * 100)}%` as `${number}%`,
      label: `${item.label} · ${Math.round(item.confidence * 100)}%`,
    })),
  } : fallback;

  const aslState = signContext.aslState;
  const spelledText = aslState?.spelledText || '';
  const currentLetter = aslState?.currentLetter || '';
  const currentConfidence = aslState?.currentConfidence || 0;
  const offlinePhrase = resolveOfflinePhrase(spelledText);
  const wordState = signContext.wordState;
  const recognizedWordPhrase = wordState?.phrase || '';
  const recognizedWord = wordState?.currentWord || '';

  const formattedWordPhrase = recognizedWordPhrase
    ? recognizedWordPhrase.split(' ').map(formatAslWord).join(' ')
    : '';

  const top3Summary = wordState?.top3 && wordState.top3.length > 0 && wordState.top3[0].confidence >= 0.20
    ? wordState.top3.map((p) => `${formatAslWord(p.label)} ${Math.round(p.confidence * 100)}%`).join(' · ')
    : '';

  const browserSignHeading = signContext.error
    ? 'Hand tracking needs attention.'
    : formattedWordPhrase
      ? formattedWordPhrase
      : recognizedWord && (wordState?.currentConfidence || 0) >= 0.30
        ? `Sign: ${formatAslWord(recognizedWord)} · ${Math.round((wordState?.currentConfidence || 0) * 100)}%`
    : offlinePhrase
      ? offlinePhrase.speech
      : spelledText
        ? `Spelled: "${spelledText}"`
      : currentLetter
        ? `Letter ${currentLetter} · ${Math.round(currentConfidence * 100)}% Match`
        : translation || (isAnalysingContext ? 'Interpreting your signed context…' : 'Ready for ASL gestures');

  const browserSignDetail = signContext.error
    ? signContext.error
    : formattedWordPhrase
      ? 'ASL dynamic sign phrase · Brief pause between signs'
      : top3Summary
        ? `Detected sign candidate: ${top3Summary}`
      : recognizedWord
        ? `Analyzing motion · ${Math.round((wordState?.bufferProgress || 0) * 100)}% temporal buffer`
    : offlinePhrase
      ? `Offline phrase · ${offlinePhrase.trigger} · no network required`
      : spelledText
        ? `Fingerspell one of ${offlinePhraseCount} offline phrases or any word · On-device model (${aslState?.accuracyScore || 88}% test accuracy)`
      : currentLetter
        ? `Hold steady to lock "${currentLetter}" into word buffer`
        : signContext.handCount > 0
          ? `${signContext.handCount} ${signContext.handCount === 1 ? 'hand' : 'hands'} tracked · Dynamic signs (Hello, Yes, No, Please, Thank you) or Fingerspell (A–Z)`
          : 'Hold hand in front of camera to sign or fingerspell';

  const signHeading = usesBrowserHandTracking
    ? browserSignHeading
    : vision?.summary || (visionState === 'offline' ? 'Sign interpretation is reconnecting…' : 'Looking for a signed message…');
  const signDetail = usesBrowserHandTracking
    ? browserSignDetail
    : vision?.speech || 'Keep both hands clearly in the camera frame.';

  const speak = (text = current.speech) => {
    if (!speaking || !text.trim()) return;
    Speech.stop();
    Speech.speak(text, { language: 'en-US', rate: 0.92 });
  };

  const switchMode = (nextMode: AssistMode) => {
    if (nextMode === mode) return;
    Speech.stop();
    setMode(nextMode);
    setSignContext(emptySignContext);
    setTranslation('');
    setTranslationError('');
    setIsAnalysingContext(false);
    translationInFlight.current = false;
    lastTranslation.current = '';
  };

  const restartContextCapture = () => {
    if (!usesBrowserHandTracking) {
      scanNow();
      return;
    }
    aslEngineRef.current?.clearText();
    setSignContext(emptySignContext);
    capturedSequence.current = [];
    capturedFrames.current = [];
    setTranslation('');
    setTranslationError('');
    setIsAnalysingContext(false);
    translationInFlight.current = false;
    lastTranslation.current = '';
    setCaptureVersion((value) => value + 1);
  };

  const interpretLiveContext = async (sequence: number[][], frames: string[]) => {
    capturedSequence.current = sequence;
    capturedFrames.current = frames;
    // Only attempt cloud translation if backend is configured
    if (frames.length !== 3 || translationInFlight.current || !hasVisionBackend) return;
    translationInFlight.current = true;
    setIsAnalysingContext(true);
    setTranslationError('');
    try {
      const interpreted = await interpretSignSequence(frames, lastTranslation.current);
      setTranslation(interpreted.transcript);
      if (interpreted.should_speak && interpreted.segment && interpreted.transcript !== lastTranslation.current) {
        speak(interpreted.segment);
      }
      lastTranslation.current = interpreted.transcript;
    } catch (error) {
      if (hasVisionBackend) {
        setTranslationError(error instanceof Error ? error.message : 'Live sign interpretation failed.');
      }
    } finally {
      setIsAnalysingContext(false);
      translationInFlight.current = false;
    }
  };

  useEffect(() => () => { void Speech.stop(); }, []);

  useEffect(() => {
    if (!usesBrowserHandTracking && isSignMode && vision?.should_speak && vision.speech) {
      if (vision.scene_key !== lastTranslation.current) {
        lastTranslation.current = vision.scene_key;
        speak(vision.speech);
      }
    }
  }, [isSignMode, usesBrowserHandTracking, vision]);

  useEffect(() => {
    if (!offlinePhrase) {
      lastOfflinePhrase.current = '';
      return;
    }
    if (offlinePhrase.trigger !== lastOfflinePhrase.current) {
      lastOfflinePhrase.current = offlinePhrase.trigger;
      speak(offlinePhrase.speech);
    }
  }, [offlinePhrase?.trigger]);

  useEffect(() => {
    if (!recognizedWordPhrase || recognizedWordPhrase === lastOfflineWordPhrase.current) return;
    const previous = lastOfflineWordPhrase.current;
    lastOfflineWordPhrase.current = recognizedWordPhrase;
    const addition = recognizedWordPhrase.slice(previous.length).trim();
    if (addition) speak(formatAslWord(addition));
  }, [recognizedWordPhrase]);

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
      {usesBrowserHandTracking ? (
        <HandTracker
          key={captureVersion}
          aslEngineRef={aslEngineRef}
          onContextChange={setSignContext}
          onContextCaptured={interpretLiveContext}
        />
      ) : (
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" zoom={0} enableTorch={torchEnabled} />
      )}
      <View style={c.shade} />
      <SafeAreaView style={c.overlay}>
        <View style={c.top}>
          <Pressable accessibilityLabel="Go back" onPress={() => router.back()} style={c.round}>
            <Ionicons name="chevron-back" size={24} color="#fff" />
          </Pressable>
          <View style={c.modeIndicator}>
            <View style={c.greenDot} />
            <Text style={c.modeText}>
              {usesBrowserHandTracking
                ? hasVisionBackend
                  ? 'HYBRID ASL · OFFLINE + CLOUD'
                  : 'OFFLINE ASL FINGERSPELLING'
                : visionState === 'offline'
                  ? 'VISION · RECONNECTING'
                  : visionState === 'analyzing'
                    ? 'VISION · ANALYZING'
                    : isSignMode
                      ? 'SIGN INTERPRETATION · LIVE'
                      : hasVisionBackend
                        ? 'HYBRID VISION · LIVE'
                        : 'LOCAL YOLO · LIVE'}
            </Text>
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

          {usesBrowserHandTracking && isSignMode && (
            <View style={c.aslHudContainer}>
              {currentLetter ? (
                <View style={c.aslLetterCard}>
                  <View style={c.aslBadge}>
                    <Text style={c.aslBadgeLetter}>{currentLetter}</Text>
                  </View>
                  <View style={c.aslMeta}>
                    <View style={c.aslMetaRow}>
                      <Text style={c.aslConfidenceText}>{Math.round(currentConfidence * 100)}% match</Text>
                      <Text style={c.aslChipText}>On-Device AI</Text>
                    </View>
                    <View style={c.aslProgressTrack}>
                      <View
                        style={[
                          c.aslProgressFill,
                          { width: `${Math.round((aslState?.holdProgress || 0) * 100)}%` },
                          aslState?.isHolding && c.aslProgressFillDone,
                        ]}
                      />
                    </View>
                  </View>
                </View>
              ) : (
                <View style={c.aslPromptCard}>
                  <Ionicons name="hand-left-outline" size={17} color="#C8FF72" />
                  <Text style={c.aslPromptText}>
                    {signContext.handCount > 0 ? 'Hold sign steady to commit' : 'Show hand in frame to spell ASL'}
                  </Text>
                </View>
              )}

            </View>
          )}
        </View>

        <View style={c.bottom}>
          {usesBrowserHandTracking ? (
            <View style={[c.aslWordCard, c.aslWordCardPanel]}>
              <View style={c.aslWordHeader}>
                <Text style={c.aslWordLabel}>OFFLINE TRANSCRIPT</Text>
                <Pressable
                  onPress={() => speak(recognizedWordPhrase || offlinePhrase?.speech || spelledText)}
                  disabled={!recognizedWordPhrase && !spelledText}
                  style={[c.aslSpeakMiniBtn, !recognizedWordPhrase && !spelledText && c.aslSpeakMiniBtnDisabled]}
                  accessibilityLabel="Speak word buffer"
                >
                  <Ionicons name="volume-high" size={15} color="#07110F" />
                  <Text style={c.aslSpeakMiniText}>Speak</Text>
                </Pressable>
              </View>
              <Text style={[c.aslWordText, !spelledText && c.aslWordTextEmpty]}>
                {recognizedWordPhrase || spelledText || 'Start signing'}
                <Text style={c.aslCursor}>_</Text>
              </Text>
              <View style={c.aslWordActions}>
                <Pressable onPress={() => aslEngineRef.current?.addSpace()} style={c.aslActionBtn} accessibilityLabel="Add space">
                  <Ionicons name="return-down-forward-outline" size={16} color="#DDE9E4" />
                  <Text style={c.aslActionText}>Space</Text>
                </Pressable>
                <Pressable onPress={() => aslEngineRef.current?.backspace()} style={c.aslActionBtn} accessibilityLabel="Delete last letter">
                  <Ionicons name="backspace-outline" size={16} color="#DDE9E4" />
                  <Text style={c.aslActionText}>Del</Text>
                </Pressable>
                <Pressable onPress={() => aslEngineRef.current?.clearText()} style={[c.aslActionBtn, c.aslActionClearBtn]} accessibilityLabel="Clear word buffer">
                  <Ionicons name="trash-outline" size={15} color="#FF826F" />
                  <Text style={[c.aslActionText, c.aslActionClearText]}>Clear</Text>
                </Pressable>
              </View>
            </View>
          ) : (
            <View style={[c.answer, isSignMode && c.signAnswer]}>
              <View style={[c.answerIcon, isSignMode && c.signAnswerIcon]}><Ionicons name={isSignMode ? 'hand-left' : current.icon} size={25} color="#07110F" /></View>
              <View style={c.answerCopy}>
                <Text style={isSignMode ? c.translationText : c.answerTitle}>{isSignMode ? signHeading : current.title}</Text>
                <Text style={isSignMode ? c.translationDetail : c.answerText}>{isSignMode ? signDetail : current.detail}</Text>
              </View>
            </View>
          )}

          {!usesBrowserHandTracking && (
            <>
              <View style={c.controls}>
                <Pressable onPress={restartContextCapture} style={c.control}>
                  <Ionicons name="refresh" size={22} color="#EAF6F0" />
                  <Text style={c.controlText}>{isSignMode ? 'Read again' : 'Scan again'}</Text>
                </Pressable>
                <Pressable accessibilityLabel={isSignMode ? 'Speak translation' : 'Repeat description'} onPress={() => speak(translation || current.speech)} style={c.speak}>
                  <Ionicons name="volume-high" size={24} color="#07110F" />
                </Pressable>
                <Pressable accessibilityLabel={torchEnabled ? 'Turn flashlight off' : 'Turn flashlight on'} onPress={() => setTorchEnabled((value) => !value)} style={c.control}>
                  <Ionicons name={torchEnabled ? 'flash' : 'flash-off'} size={22} color={torchEnabled ? '#C8FF72' : '#EAF6F0'} />
                  <Text style={[c.controlText, torchEnabled && c.controlTextActive]}>{torchEnabled ? 'Flash on' : 'Flashlight'}</Text>
                </Pressable>
              </View>
              <Text style={c.hint}>{isSignMode ? 'Tap the megaphone to speak the interpreted text aloud' : 'Tap the speaker button to repeat the description'}</Text>
            </>
          )}
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

  // On-Device ASL Live HUD Styling
  aslHudContainer: {
    position: 'absolute',
    top: 6,
    left: 0,
    right: 0,
    alignItems: 'center',
    gap: 10,
    zIndex: 10,
  },
  aslLetterCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: 'rgba(7, 24, 18, 0.94)',
    borderWidth: 1.5,
    borderColor: '#C8FF72',
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 9,
    shadowColor: '#C8FF72',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 10,
  },
  aslBadge: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#C8FF72',
    alignItems: 'center',
    justifyContent: 'center',
  },
  aslBadgeLetter: {
    color: '#07110F',
    fontSize: 26,
    fontWeight: '900',
  },
  aslMeta: {
    minWidth: 130,
    gap: 5,
  },
  aslMetaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 6,
  },
  aslConfidenceText: {
    color: '#F2FAF6',
    fontSize: 12,
    fontWeight: '800',
  },
  aslChipText: {
    color: '#C8FF72',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.8,
    backgroundColor: 'rgba(200, 255, 114, 0.16)',
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 5,
  },
  aslProgressTrack: {
    height: 5,
    backgroundColor: 'rgba(255, 255, 255, 0.14)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  aslProgressFill: {
    height: '100%',
    backgroundColor: '#95E344',
    borderRadius: 3,
  },
  aslProgressFillDone: {
    backgroundColor: '#C8FF72',
  },
  aslPromptCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(5, 18, 14, 0.88)',
    borderWidth: 1,
    borderColor: 'rgba(200, 255, 114, 0.35)',
    borderRadius: 99,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  aslPromptText: {
    color: '#DCECE5',
    fontSize: 11,
    fontWeight: '700',
  },
  aslWordCard: {
    backgroundColor: 'rgba(5, 20, 15, 0.95)',
    borderWidth: 1,
    borderColor: 'rgba(200, 255, 114, 0.45)',
    borderRadius: 16,
    padding: 12,
    width: '100%',
    alignSelf: 'center',
  },
  aslWordCardPanel: {
    width: '100%',
    padding: 14,
    borderRadius: 20,
    borderColor: 'rgba(200, 255, 114, 0.62)',
    backgroundColor: 'rgba(5, 26, 17, 0.95)',
  },
  aslWordHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  aslWordLabel: {
    color: '#769286',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  aslSpeakMiniBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#C8FF72',
    borderRadius: 9,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  aslSpeakMiniBtnDisabled: { opacity: 0.45 },
  aslSpeakMiniText: {
    color: '#07110F',
    fontSize: 11,
    fontWeight: '800',
  },
  aslWordText: {
    color: '#F7FBF8',
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: 1.2,
    marginVertical: 8,
    minHeight: 32,
  },
  aslWordTextEmpty: { color: '#8DA39A', fontWeight: '700' },
  aslCursor: {
    color: '#C8FF72',
    fontWeight: '400',
  },
  aslWordActions: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.08)',
    paddingTop: 9,
  },
  aslActionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 10,
    paddingVertical: 9,
  },
  aslActionClearBtn: {
    backgroundColor: 'rgba(255, 130, 111, 0.12)',
  },
  aslActionText: {
    color: '#DCEAE4',
    fontSize: 11,
    fontWeight: '700',
  },
  aslActionClearText: { color: '#FF826F' },
});

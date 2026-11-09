import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router, useLocalSearchParams } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import HandTracker, { type SignContextState } from '../components/HandTracker';
import {
  checkWhatChanged,
  getCampusAlerts,
  type CampusAlert,
  type WhatChangedResult,
} from '../src/campusMemory';
import { resolveOfflinePhrase } from '../src/offlinePhrasebook';
import { AssistMode, getApiUrl, hasVisionBackend, interpretSignSequence, useVisionLoop } from '../src/vision';

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

export default function CameraScreen() {
  const { initialMode } = useLocalSearchParams<{ initialMode?: string }>();
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<AssistMode>(initialMode === 'sign' ? 'sign' : 'scene');
  const [speaking, setSpeaking] = useState(true);
  const [torchEnabled, setTorchEnabled] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [signContext, setSignContext] = useState<SignContextState>(emptySignContext);
  const [isAnalysingContext, setIsAnalysingContext] = useState(false);
  const [translation, setTranslation] = useState('');
  const [translationError, setTranslationError] = useState('');
  const [captureVersion, setCaptureVersion] = useState(0);
  const [campusLocation] = useState(() => (typeof window !== 'undefined' && window.localStorage ? window.localStorage.getItem('guidelens_campus_location') || 'Library Corridor' : 'Library Corridor'));
  const [activeAlert, setActiveAlert] = useState<CampusAlert | null>(null);
  const [deltaNotice, setDeltaNotice] = useState<string>('');
  const [isCheckingDelta, setIsCheckingDelta] = useState(false);
  const hasSpokenAlert = useRef(false);
  const translationInFlight = useRef(false);
  const lastTranslation = useRef('');
  const lastApproachWarning = useRef('');
  const lastObstacleWarning = useRef('');
  const lastRecognizedWord = useRef('');
  const lastSpokenWordTimestamp = useRef(0);
  const wordAbsenceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const capturedSequence = useRef<number[][]>([]);
  const capturedFrames = useRef<string[]>([]);
  const cameraRef = useRef<CameraView>(null);
  const aslEngineRef = useRef<{
    backspace: () => string;
    addSpace: () => string;
    clearText: () => void;
  } | null>(null);

  const isSignMode = mode === 'sign';
  const showVisionDebug = process.env.EXPO_PUBLIC_VISION_DEBUG === 'true';
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
    ? `Detected ${detections.slice(0, 3).map((item) => `${item.label}${item.direction ? ` ${item.direction}` : ''}`).join(', ')}`
    : 'Looking for nearby objects…';
  const serverBoxes = (vision?.objects || [])
    .filter((item) => [item.x, item.y, item.width, item.height].every((value) => typeof value === 'number'))
    .map((item) => ({
      l: `${Math.round((item.x || 0) * 100)}%` as `${number}%`,
      t: `${Math.round((item.y || 0) * 100)}%` as `${number}%`,
      w: `${Math.round((item.width || 0) * 100)}%` as `${number}%`,
      h: `${Math.round((item.height || 0) * 100)}%` as `${number}%`,
      label: `${item.label}${item.distance_m ? ` · ~${item.distance_m.toFixed(1)}m` : item.relative_proximity != null ? ` · ${item.relative_proximity >= 0.67 ? 'nearer' : item.relative_proximity <= 0.33 ? 'farther' : 'mid-depth'}` : ` · ${Math.round(item.confidence * 100)}%`}${showVisionDebug ? ` · id ${item.track_id ?? '—'} · ${item.motion?.direction || item.position} · ${item.motion?.trend || 'unknown'} · ${Math.round((item.motion?.confidence ?? item.confidence) * 100)}%` : ''}`,
    }));
  const current: AssistFrame = vision || detections.length ? {
    title: vision?.summary || localSummary,
    detail: vision?.speech || 'Real-time spatial tracker running privately on this device.',
    speech: vision?.speech || localSummary,
    icon: mode === 'sign' ? 'hand-left' : vision?.urgency === 'high' ? 'warning' : 'navigate',
    // Prefer server YOLO boxes when supplied: they are the detections paired
    // with metric depth. Keep local YOLOX as a fast offline fallback.
    boxes: (serverBoxes.length ? serverBoxes : detections.map((item) => ({
      l: `${Math.round(item.x * 100)}%` as `${number}%`,
      t: `${Math.round(item.y * 100)}%` as `${number}%`,
      w: `${Math.round(item.width * 100)}%` as `${number}%`,
      h: `${Math.round(item.height * 100)}%` as `${number}%`,
      label: `${item.label}${item.direction ? ` · ${item.direction}` : ''}`,
    }))),
  } : fallback;

  // Fingerspelling and isolated-sign classifiers are capture aids only. Do
  // not present their noisy letter/word guesses as a contextual translation.
  // Never fall back to isolated letters/words here. The board represents only
  // the contextual decoder, but exposes its real state so a stalled request
  // is not mistaken for an empty translation.
  const displayedTranscript = translation;

  const browserSignHeading = signContext.error
    ? 'Hand tracking needs attention.'
    // A local word recognizer is useful as a capture aid, but it must never
    // override the sentence-level result from the contextual decoder.
    : translation
      ? translation
      : isAnalysingContext
        ? 'Analyzing the signed phrase…'
        : 'Ready for a complete signed phrase';

  const browserSignDetail = signContext.error
    ? signContext.error
    : translation
      ? 'Contextual phrase result · Confirm its meaning with the signer'
      : isAnalysingContext
        ? 'Collecting an overlapping sequence before producing a phrase result'
        : signContext.handCount > 0
              ? `${signContext.handCount} ${signContext.handCount === 1 ? 'hand' : 'hands'} · ${signContext.faceTracked ? 'face' : 'face searching'} · ${signContext.poseTracked ? 'body pose' : 'body searching'} · multimodal movement capture`
              : 'Frame your hands, face, and upper body to sign';

  const offlineWord = signContext.wordState?.currentWord;

  const contextBoardText = displayedTranscript
    || (offlineWord ? `Sign: "${offlineWord.toUpperCase()}" (On-device)` : '')
    || (isAnalysingContext
      ? 'Analyzing signed gesture…'
      : signContext.ready
        ? 'Ready for next sign (On-device)…'
        : signContext.handCount > 0
          ? `${signContext.handCount} ${signContext.handCount === 1 ? 'hand' : 'hands'} tracked · live capture`
          : 'Show your hands to begin signing');
  const contextBoardIsError = false;

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
    if (frames.length !== 3 || translationInFlight.current) return;
    const apiUrl = getApiUrl();
    if (!apiUrl) return;

    translationInFlight.current = true;
    setIsAnalysingContext(true);
    setTranslationError('');
    const requestController = new AbortController();
    const timeout = setTimeout(() => requestController.abort(), 6_000);
    try {
      const interpreted = await interpretSignSequence(frames, lastTranslation.current, requestController.signal);
      if (interpreted && interpreted.transcript) {
        setTranslation(interpreted.transcript);
        if (interpreted.should_speak && interpreted.segment && interpreted.transcript !== lastTranslation.current) {
          speak(interpreted.segment);
        }
        lastTranslation.current = interpreted.transcript;
      }
    } catch {
      // Cloud is offline or unreachable: silently continue on-device without any error!
    } finally {
      clearTimeout(timeout);
      setIsAnalysingContext(false);
      translationInFlight.current = false;
    }
  };

  useEffect(() => () => {
    void Speech.stop();
    if (wordAbsenceTimer.current) clearTimeout(wordAbsenceTimer.current);
  }, []);

  useEffect(() => {
    if (!usesBrowserHandTracking && isSignMode && vision?.should_speak && vision.speech) {
      if (vision.scene_key !== lastTranslation.current) {
        lastTranslation.current = vision.scene_key;
        speak(vision.speech);
      }
    }
  }, [isSignMode, usesBrowserHandTracking, vision]);

  useEffect(() => {
    if (isSignMode || !vision?.approaching?.length) return;
    const warning = vision.approaching.map((item) => `${item.label}:${item.distance_m}`).join('|');
    if (warning !== lastApproachWarning.current) {
      lastApproachWarning.current = warning;
      speak(vision.speech);
    }
  }, [isSignMode, vision?.approaching, vision?.speech]);

  useEffect(() => {
    if (isSignMode || !vision?.should_speak || !vision.speech || vision.urgency === 'low') return;
    if (vision.scene_key !== lastObstacleWarning.current) {
      lastObstacleWarning.current = vision.scene_key;
      speak(vision.speech);
    }
  }, [isSignMode, vision?.should_speak, vision?.speech, vision?.urgency, vision?.scene_key]);

  useEffect(() => {
    if (!offlineWord || translation) {
      if (!wordAbsenceTimer.current && lastRecognizedWord.current) {
        wordAbsenceTimer.current = setTimeout(() => {
          lastRecognizedWord.current = '';
          wordAbsenceTimer.current = null;
        }, 1500);
      }
      return;
    }

    if (wordAbsenceTimer.current) {
      clearTimeout(wordAbsenceTimer.current);
      wordAbsenceTimer.current = null;
    }

    const now = Date.now();
    const isDifferentWord = offlineWord !== lastRecognizedWord.current;
    const cooldownElapsed = now - lastSpokenWordTimestamp.current > 3200;

    if (isDifferentWord || cooldownElapsed) {
      lastRecognizedWord.current = offlineWord;
      lastSpokenWordTimestamp.current = now;
      const phrase = resolveOfflinePhrase(offlineWord);
      speak(phrase ? phrase.speech : offlineWord);
    }
  }, [offlineWord, translation]);

  useEffect(() => {
    if (mode !== 'scene') return;
    let isMounted = true;
    void getCampusAlerts(campusLocation).then((alerts) => {
      if (!isMounted || alerts.length === 0) return;
      const highest = alerts[0];
      setActiveAlert(highest);
      if (highest && !hasSpokenAlert.current && highest.description) {
        hasSpokenAlert.current = true;
        Speech.speak(`Campus alert for ${highest.location}: ${highest.description}`, { rate: 0.95 });
      }
    });
    return () => { isMounted = false; };
  }, [mode, campusLocation]);

  const handleWhatChanged = async () => {
    Speech.stop();
    setIsCheckingDelta(true);
    try {
      const res = await checkWhatChanged(campusLocation, detections);
      setDeltaNotice(res.delta_summary);
      Speech.speak(res.speech, { rate: 0.95 });
    } catch {
      setDeltaNotice('Baseline checked. No changes detected.');
    } finally {
      setIsCheckingDelta(false);
    }
  };

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
          showLandmarks={showLandmarks}
          onContextChange={setSignContext}
          onContextCaptured={interpretLiveContext}
        />
      ) : (
        <>
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" zoom={0} enableTorch={torchEnabled} />
          <View style={c.boxesLayer} pointerEvents="none">
            {current.boxes.map((box, index) => (
              <View key={`${mode}-${index}`} style={[c.box, { left: box.l, top: box.t, width: box.w, height: box.h }, isSignMode && c.signBox]}>
                <View style={c.boxLabelContainer}>
                  <Text style={c.boxLabel}>{box.label}</Text>
                </View>
              </View>
            ))}
          </View>
        </>
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
                ? 'ON-DEVICE ASL · OFFLINE READY'
                : visionState === 'offline'
                  ? 'ON-DEVICE VISION · OFFLINE READY'
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

        {activeAlert && mode === 'scene' && (
          <View style={c.campusAlertBar}>
            <View style={c.alertIconBadge}>
              <Ionicons name="warning" size={13} color="#07110F" />
            </View>
            <View style={c.alertTextGroup}>
              <Text style={c.alertLoc}>{activeAlert.location.toUpperCase()} · {activeAlert.relative_time}:</Text>
              <Text style={c.alertDesc} numberOfLines={1}>{activeAlert.description}</Text>
            </View>
          </View>
        )}

        <View style={c.modeTabs} accessibilityRole="tablist">
          <Pressable accessibilityRole="tab" accessibilityState={{ selected: mode === 'scene' }} onPress={() => switchMode('scene')} style={[c.modeTab, mode === 'scene' && c.modeTabActive]}>
            <Ionicons name="eye-outline" size={17} color={mode === 'scene' ? '#09130D' : '#DDE9E4'} />
            <Text style={[c.modeTabText, mode === 'scene' && c.modeTabTextActive]}>Scene assist</Text>
          </Pressable>
          <Pressable accessibilityRole="tab" accessibilityState={{ selected: mode === 'sign' }} onPress={() => switchMode('sign')} style={[c.modeTab, mode === 'sign' && c.modeTabActive]}>
            <Ionicons name="hand-left-outline" size={17} color={mode === 'sign' ? '#09130D' : '#DDE9E4'} />
            <Text style={[c.modeTabText, mode === 'sign' && c.modeTabTextActive]}>Sign to voice</Text>
          </Pressable>
          <Pressable accessibilityRole="tab" onPress={() => router.push('/bridge' as any)} style={c.modeTab}>
            <Ionicons name="people-outline" size={17} color="#DDE9E4" />
            <Text style={c.modeTabText}>Bridge</Text>
          </Pressable>
        </View>

        <View style={c.reticle}>
          {usesBrowserHandTracking && isSignMode && (
            <View style={c.aslHudContainer}>
              <View style={c.aslPromptCard}>
                <Ionicons name="chatbubbles-outline" size={17} color="#C8FF72" />
                <Text style={c.aslPromptText}>Context-aware sign translation</Text>
              </View>
            </View>
          )}
        </View>

        <View style={c.bottom}>
          {usesBrowserHandTracking ? (
            <View style={[c.aslWordCard, c.aslWordCardPanel]}>
              <View style={c.aslWordHeader}>
                <Text style={c.aslWordLabel}>SIGN CONTEXT</Text>
                <View style={c.aslHeaderActions}>
                  <Pressable
                    onPress={() => setShowLandmarks((visible) => !visible)}
                    style={c.aslOverlayToggle}
                    accessibilityLabel={showLandmarks ? 'Hide visual tracking lines' : 'Show visual tracking lines'}
                  >
                    <Ionicons name={showLandmarks ? 'eye-off-outline' : 'eye-outline'} size={15} color="#DDE9E4" />
                    <Text style={c.aslOverlayToggleText}>{showLandmarks ? 'Hide lines' : 'Show lines'}</Text>
                  </Pressable>
                  <Pressable
                    onPress={() => speak(displayedTranscript)}
                    disabled={!displayedTranscript}
                    style={[c.aslSpeakMiniBtn, !displayedTranscript && c.aslSpeakMiniBtnDisabled]}
                    accessibilityLabel="Speak contextual translation"
                  >
                    <Ionicons name="volume-high" size={15} color="#07110F" />
                    <Text style={c.aslSpeakMiniText}>Speak</Text>
                  </Pressable>
                </View>
              </View>
              <Text style={[c.aslWordText, !displayedTranscript && c.aslWordTextEmpty, contextBoardIsError && c.aslWordTextError]}>
                {contextBoardText}
                <Text style={c.aslCursor}>_</Text>
              </Text>
              <View style={c.aslWordActions}>
                <Pressable onPress={restartContextCapture} style={[c.aslActionBtn, c.aslActionClearBtn]} accessibilityLabel="Clear sign context">
                  <Ionicons name="trash-outline" size={15} color="#FF826F" />
                  <Text style={[c.aslActionText, c.aslActionClearText]}>Clear context</Text>
                </Pressable>
              </View>
            </View>
          ) : !usesBrowserHandTracking ? (
            <View style={[c.answer, isSignMode && c.signAnswer]}>
              <View style={[c.answerIcon, isSignMode && c.signAnswerIcon]}><Ionicons name={isSignMode ? 'hand-left' : current.icon} size={25} color="#07110F" /></View>
              <View style={c.answerCopy}>
                <Text style={isSignMode ? c.translationText : c.answerTitle}>{isSignMode ? signHeading : current.title}</Text>
                <Text style={isSignMode ? c.translationDetail : c.answerText}>{isSignMode ? signDetail : current.detail}</Text>
              </View>
            </View>
          ) : null}

          {!usesBrowserHandTracking && deltaNotice ? (
            <View style={c.deltaNoticeCard}>
              <View style={c.deltaIconGroup}>
                <Ionicons name="git-compare" size={16} color="#07110F" />
                <Text style={c.deltaNoticeLabel}>DELTA (WHAT CHANGED):</Text>
              </View>
              <Text style={c.deltaNoticeText}>{deltaNotice}</Text>
              <Pressable onPress={() => setDeltaNotice('')} style={c.deltaClose}>
                <Ionicons name="close" size={14} color="#07110F" />
              </Pressable>
            </View>
          ) : null}

          {!usesBrowserHandTracking && (
            <>
              <View style={c.controls}>
                <Pressable onPress={restartContextCapture} style={c.control}>
                  <Ionicons name="refresh" size={22} color="#EAF6F0" />
                  <Text style={c.controlText}>{isSignMode ? 'Read again' : 'Scan again'}</Text>
                </Pressable>
                {!isSignMode && (
                  <Pressable
                    onPress={handleWhatChanged}
                    style={[c.control, isCheckingDelta && c.controlActive]}
                    accessibilityLabel="Check what changed in this room"
                  >
                    <Ionicons
                      name="git-compare-outline"
                      size={22}
                      color={isCheckingDelta ? '#07110F' : '#B7FF56'}
                    />
                    <Text style={[c.controlText, c.deltaControlText]}>
                      {isCheckingDelta ? 'Checking…' : 'Changed?'}
                    </Text>
                  </Pressable>
                )}
                <Pressable accessibilityLabel={isSignMode ? 'Speak translation' : 'Repeat description'} onPress={() => speak(translation || current.speech)} style={c.speak}>
                  <Ionicons name="volume-high" size={24} color="#07110F" />
                </Pressable>
                <Pressable accessibilityLabel={torchEnabled ? 'Turn flashlight off' : 'Turn flashlight on'} onPress={() => setTorchEnabled((value) => !value)} style={c.control}>
                  <Ionicons name={torchEnabled ? 'flash' : 'flash-off'} size={22} color={torchEnabled ? '#C8FF72' : '#EAF6F0'} />
                  <Text style={[c.controlText, torchEnabled && c.controlTextActive]}>{torchEnabled ? 'Flash on' : 'Flashlight'}</Text>
                </Pressable>
              </View>
              <Text style={c.hint}>{isSignMode ? 'Tap the megaphone to speak the interpreted text aloud' : 'Tap Changed? to hear only what is new, or the speaker to repeat'}</Text>
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
  boxesLayer: { ...StyleSheet.absoluteFill, zIndex: 1 },
  reticle: { flex: 1, marginVertical: 12, position: 'relative' },
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#B7FF56',
    borderRadius: 8,
    shadowColor: '#B7FF56',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.35,
    shadowRadius: 6,
  },
  signBox: { borderColor: '#C8FF72', shadowColor: '#C8FF72' },
  boxLabelContainer: {
    position: 'absolute',
    top: -24,
    left: -2,
    backgroundColor: 'rgba(7, 24, 18, 0.94)',
    borderWidth: 1,
    borderColor: '#B7FF56',
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 3,
  },
  boxLabel: {
    color: '#B7FF56',
    fontWeight: '800',
    fontSize: 10,
    letterSpacing: 0.3,
  },
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
  aslHeaderActions: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  aslOverlayToggle: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: 'rgba(221,233,228,.12)', borderRadius: 9,
    paddingHorizontal: 9, paddingVertical: 6,
  },
  aslOverlayToggleText: { color: '#DDE9E4', fontSize: 10, fontWeight: '800' },
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
  aslWordTextError: { color: '#FFB0A2', fontSize: 15, lineHeight: 21 },
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
  suggestionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginVertical: 6,
  },
  suggestionChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#C8FF72',
    paddingHorizontal: 9,
    paddingVertical: 5,
    borderRadius: 8,
  },
  suggestionChipText: {
    color: '#07110F',
    fontSize: 11,
    fontWeight: '800',
  },

  // Shared Sight & What Changed Styling
  campusAlertBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#FFD043',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginTop: 10,
    shadowColor: '#FFD043',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.35,
    shadowRadius: 6,
  },
  alertIconBadge: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(7, 17, 15, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  alertTextGroup: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  alertLoc: {
    color: '#07110F',
    fontSize: 11,
    fontWeight: '900',
  },
  alertDesc: {
    color: '#242111',
    fontSize: 11,
    fontWeight: '600',
    flex: 1,
  },
  deltaNoticeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#B7FF56',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginBottom: 10,
    shadowColor: '#B7FF56',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.35,
    shadowRadius: 8,
  },
  deltaIconGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  deltaNoticeLabel: {
    color: '#07110F',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  deltaNoticeText: {
    color: '#07110F',
    fontSize: 12,
    fontWeight: '700',
    flex: 1,
  },
  deltaClose: {
    padding: 4,
  },
  deltaControlText: {
    color: '#B7FF56',
  },
  controlActive: {
    backgroundColor: '#B7FF56',
  },
});

import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import {
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import HandTracker, { type SignContextState } from '../components/HandTracker';
import {
  getBridgeMessages,
  sendBridgeMessage,
  type BridgeMessage,
} from '../src/campusMemory';

const QUICK_DEAF_SIGNS = [
  'THANK YOU',
  'HELP',
  'WHERE IS ROOM 204?',
  'HELLO',
  'YES',
  'NO',
  'PLEASE WAIT',
  'WATER',
];

const QUICK_BLIND_REPLIES = [
  'I can help you, follow me.',
  'Room 204 is down the hall on your right.',
  'Please wait a moment.',
  "You're welcome! Glad to help.",
  'Yes, I hear you.',
  'No problem at all.',
];

export default function DeafBlindBridgeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [sessionId] = useState('campus-bridge-session');
  const [messages, setMessages] = useState<BridgeMessage[]>([]);
  const [customReply, setCustomReply] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastDeafSign, setLastDeafSign] = useState<string>('');
  const [lastBlindReply, setLastBlindReply] = useState<string>('');
  const [cameraFacing, setCameraFacing] = useState<'front' | 'back'>('front');
  const [isListeningMic, setIsListeningMic] = useState(false);

  const lastSpokenText = useRef<string>('');
  const scrollViewRef = useRef<any>(null);

  // Load initial conversation history
  useEffect(() => {
    void getBridgeMessages(sessionId).then((msgs) => {
      setMessages(msgs);
      if (msgs.length > 0) {
        const last = msgs[msgs.length - 1];
        if (last.sender_role === 'deaf') setLastDeafSign(last.text);
        if (last.sender_role === 'blind') setLastBlindReply(last.text);
      }
    });
  }, [sessionId]);

  // Request camera permission if not granted
  useEffect(() => {
    if (permission && !permission.granted && permission.canAskAgain) {
      void requestPermission();
    }
  }, [permission, requestPermission]);

  // Handle hand landmark sign context changes from HandTracker
  const handleContextChange = (state: SignContextState) => {
    // If a confident word was recognized from the sign engine
    const detectedWord =
      state.wordState?.currentWord || state.aslState?.currentLetter;
    if (
      detectedWord &&
      detectedWord !== lastSpokenText.current &&
      (state.wordState?.currentConfidence ?? 0) >= 0.70
    ) {
      lastSpokenText.current = detectedWord;
      void handleDeafSigned(detectedWord.toUpperCase());
    }
  };

  const handleDeafSigned = async (text: string) => {
    if (!text.trim()) return;
    setLastDeafSign(text);
    setIsSpeaking(true);

    // 1. Speak aloud for the blind student
    Speech.speak(text, {
      rate: 0.95,
      pitch: 1.0,
      onDone: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });

    // 2. Post to bridge message store
    const msg = await sendBridgeMessage(sessionId, 'deaf', text);
    setMessages((prev) => [...prev, msg]);
    scrollViewRef.current?.scrollToEnd({ animated: true });
  };

  const handleBlindSpokenReply = async (text: string) => {
    if (!text.trim()) return;
    setLastBlindReply(text);
    setCustomReply('');

    // 1. Speak confirmation chime or feedback
    Speech.speak(`Displaying to deaf partner: ${text}`, {
      rate: 1.1,
    });

    // 2. Post to bridge message store so deaf student sees it
    const msg = await sendBridgeMessage(sessionId, 'blind', text);
    setMessages((prev) => [...prev, msg]);
    scrollViewRef.current?.scrollToEnd({ animated: true });
  };

  // Browser speech recognition for mic input (Web / Electron)
  const toggleMicListening = () => {
    if (typeof window === 'undefined') return;
    const SpeechRec =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      Speech.speak('Speech recognition unavailable. Use quick reply buttons.');
      return;
    }

    if (isListeningMic) {
      setIsListeningMic(false);
      return;
    }

    try {
      const recognition = new SpeechRec();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        setIsListeningMic(true);
        Speech.speak('Listening for your reply...');
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        if (transcript) {
          void handleBlindSpokenReply(transcript);
        }
      };

      recognition.onerror = () => setIsListeningMic(false);
      recognition.onend = () => setIsListeningMic(false);

      recognition.start();
    } catch {
      setIsListeningMic(false);
    }
  };

  return (
    <SafeAreaView style={s.safe}>
      {/* Top Header */}
      <View style={s.header}>
        <Pressable
          style={s.backBtn}
          onPress={() => router.back()}
          accessibilityRole="button"
          accessibilityLabel="Back to Assistant Hub"
        >
          <Ionicons name="arrow-back" size={20} color="#FFFFFF" />
        </Pressable>

        <View style={s.headerCenter}>
          <View style={s.bridgeBadge}>
            <Ionicons name="people" size={14} color="#07110F" />
            <Text style={s.bridgeBadgeText}>DEAF–BLIND BRIDGE</Text>
          </View>
          <Text style={s.subHeader}>Two-Way Direct Conversation Aid</Text>
        </View>

        <View style={s.localBadge}>
          <View style={s.localDot} />
          <Text style={s.localText}>ON-DEVICE</Text>
        </View>
      </View>

      <ScrollView
        ref={scrollViewRef}
        style={s.scroll}
        contentContainerStyle={s.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {/* UPPER PANEL: DEAF SIGNING SECTION */}
        <View style={s.panelDeaf}>
          <View style={s.panelHeader}>
            <View style={s.roleTagDeaf}>
              <Ionicons name="hand-right" size={16} color="#B7FF56" />
              <Text style={s.roleTagTextDeaf}>DEAF STUDENT (SIGNS ASL)</Text>
            </View>
            {isSpeaking && (
              <View style={s.speakingPill}>
                <Ionicons name="volume-high" size={14} color="#07110F" />
                <Text style={s.speakingPillText}>Speaking to blind student…</Text>
              </View>
            )}
          </View>

          {/* Camera preview for signing */}
          <View style={s.cameraCard}>
            {Platform.OS === 'web' ? (
              <HandTracker
                onContextChange={handleContextChange}
                showLandmarks={true}
              />
            ) : permission?.granted ? (
              <CameraView
                facing={cameraFacing}
                style={StyleSheet.absoluteFill}
              />
            ) : (
              <View style={s.cameraFallback}>
                <Ionicons name="videocam-off" size={32} color="#607B6E" />
                <Text style={s.fallbackText}>Camera initializing…</Text>
              </View>
            )}

            <View style={s.cameraOverlayBar}>
              <Text style={s.overlayPrompt}>
                Sign with hands in frame or pick a sign:
              </Text>
              <Pressable
                style={s.flipBtn}
                onPress={() =>
                  setCameraFacing((c) => (c === 'front' ? 'back' : 'front'))
                }
              >
                <Ionicons name="camera-reverse" size={16} color="#FFFFFF" />
              </Pressable>
            </View>
          </View>

          {/* Current Signed Translation Card */}
          <View style={s.signedCard}>
            <Text style={s.signedLabel}>SPOKEN TO BLIND PARTNER:</Text>
            <Text style={s.signedText}>
              {lastDeafSign ? `"${lastDeafSign}"` : 'Waiting for a gesture…'}
            </Text>
          </View>

          {/* Quick Sign Action Chips */}
          <View style={s.chipsRow}>
            {QUICK_DEAF_SIGNS.map((sign) => (
              <Pressable
                key={sign}
                style={s.signChip}
                onPress={() => handleDeafSigned(sign)}
                accessibilityRole="button"
                accessibilityLabel={`Sign ${sign}`}
              >
                <Text style={s.signChipText}>{sign}</Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* DIVIDER */}
        <View style={s.dividerRow}>
          <View style={s.dividerLine} />
          <View style={s.dividerBadge}>
            <Ionicons name="swap-vertical" size={16} color="#B7FF56" />
          </View>
          <View style={s.dividerLine} />
        </View>

        {/* LOWER PANEL: BLIND STUDENT REPLY SECTION (HIGH CONTRAST) */}
        <View style={s.panelBlind}>
          <View style={s.panelHeader}>
            <View style={s.roleTagBlind}>
              <Ionicons name="mic" size={16} color="#FFD043" />
              <Text style={s.roleTagTextBlind}>BLIND STUDENT (VOICE REPLY)</Text>
            </View>
            <Text style={s.displayHint}>Displays large text for deaf partner</Text>
          </View>

          {/* GIANT HIGH-CONTRAST DISPLAY FOR DEAF STUDENT TO READ */}
          <View style={s.giantTextCard}>
            <Text style={s.giantTextLabel}>SHOWN TO DEAF PARTNER:</Text>
            <Text
              style={s.giantText}
              accessibilityRole="text"
              accessibilityLiveRegion="assertive"
            >
              {lastBlindReply
                ? lastBlindReply.toUpperCase()
                : 'PRESS A REPLY BELOW OR SPEAK'}
            </Text>
          </View>

          {/* Blind Student Quick Tactile Responses */}
          <Text style={s.sectionSubtitle}>
            Tactile Quick Replies (Touch to Speak & Display):
          </Text>
          <View style={s.repliesGrid}>
            {QUICK_BLIND_REPLIES.map((reply) => (
              <Pressable
                key={reply}
                style={s.replyCard}
                onPress={() => handleBlindSpokenReply(reply)}
                accessibilityRole="button"
                accessibilityLabel={reply}
              >
                <Ionicons name="volume-medium-outline" size={16} color="#FFD043" />
                <Text style={s.replyCardText}>{reply}</Text>
              </Pressable>
            ))}
          </View>

          {/* Voice Mic & Custom Text Input */}
          <View style={s.inputRow}>
            <Pressable
              style={[s.micBtn, isListeningMic && s.micBtnActive]}
              onPress={toggleMicListening}
              accessibilityRole="button"
              accessibilityLabel={isListeningMic ? 'Stop microphone' : 'Speak reply'}
            >
              <Ionicons
                name={isListeningMic ? 'mic' : 'mic-outline'}
                size={22}
                color={isListeningMic ? '#07110F' : '#FFD043'}
              />
            </Pressable>
            <TextInput
              style={s.customInput}
              value={customReply}
              onChangeText={setCustomReply}
              placeholder="Or type a custom reply…"
              placeholderTextColor="#607B6E"
              onSubmitEditing={() => handleBlindSpokenReply(customReply)}
            />
            <Pressable
              style={s.sendBtn}
              onPress={() => handleBlindSpokenReply(customReply)}
              accessibilityRole="button"
              accessibilityLabel="Send reply"
            >
              <Ionicons name="send" size={16} color="#07110F" />
            </Pressable>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#07110F',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1A2925',
  },
  backBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#11221D',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerCenter: {
    alignItems: 'center',
  },
  bridgeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#B7FF56',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 6,
  },
  bridgeBadgeText: {
    color: '#07110F',
    fontWeight: '800',
    fontSize: 12,
    letterSpacing: 0.5,
  },
  subHeader: {
    color: '#9EB5AA',
    fontSize: 11,
    marginTop: 3,
  },
  localBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0F261E',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    gap: 4,
  },
  localDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#B7FF56',
  },
  localText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '700',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  panelDeaf: {
    backgroundColor: '#0B1A16',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#1C362C',
    padding: 16,
  },
  panelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  roleTagDeaf: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  roleTagTextDeaf: {
    color: '#B7FF56',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  speakingPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#B7FF56',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
    gap: 4,
  },
  speakingPillText: {
    color: '#07110F',
    fontSize: 11,
    fontWeight: '700',
  },
  cameraCard: {
    height: 180,
    backgroundColor: '#050D0B',
    borderRadius: 12,
    overflow: 'hidden',
    position: 'relative',
    marginBottom: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraFallback: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  fallbackText: {
    color: '#607B6E',
    fontSize: 12,
    marginTop: 6,
  },
  cameraOverlayBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(7, 17, 15, 0.85)',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  overlayPrompt: {
    color: '#B2C7BC',
    fontSize: 11,
  },
  flipBtn: {
    padding: 4,
  },
  signedCard: {
    backgroundColor: '#11221D',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#B7FF56',
  },
  signedLabel: {
    color: '#769485',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  signedText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  chipsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  signChip: {
    backgroundColor: '#162C24',
    borderWidth: 1,
    borderColor: '#244537',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  signChipText: {
    color: '#B7FF56',
    fontSize: 12,
    fontWeight: '700',
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 18,
    gap: 12,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#1C362C',
  },
  dividerBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#11221D',
    borderWidth: 1,
    borderColor: '#1C362C',
    justifyContent: 'center',
    alignItems: 'center',
  },
  panelBlind: {
    backgroundColor: '#0B1A16',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#3D3112',
    padding: 16,
  },
  roleTagBlind: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  roleTagTextBlind: {
    color: '#FFD043',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  displayHint: {
    color: '#769485',
    fontSize: 10,
  },
  giantTextCard: {
    backgroundColor: '#000000',
    borderWidth: 2,
    borderColor: '#FFD043',
    borderRadius: 12,
    padding: 16,
    marginVertical: 12,
  },
  giantTextLabel: {
    color: '#8A7A42',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  giantText: {
    color: '#FFD043',
    fontSize: 26,
    fontWeight: '900',
    lineHeight: 32,
    letterSpacing: 0.5,
  },
  sectionSubtitle: {
    color: '#9EB5AA',
    fontSize: 11,
    fontWeight: '600',
    marginBottom: 8,
  },
  repliesGrid: {
    gap: 8,
    marginBottom: 14,
  },
  replyCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#172720',
    borderWidth: 1,
    borderColor: '#2B4237',
    padding: 12,
    borderRadius: 10,
    gap: 10,
  },
  replyCardText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '600',
    flex: 1,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  micBtn: {
    width: 44,
    height: 44,
    borderRadius: 10,
    backgroundColor: '#1D2A18',
    borderWidth: 1,
    borderColor: '#FFD043',
    justifyContent: 'center',
    alignItems: 'center',
  },
  micBtnActive: {
    backgroundColor: '#FFD043',
  },
  customInput: {
    flex: 1,
    height: 44,
    backgroundColor: '#11221D',
    borderRadius: 10,
    paddingHorizontal: 12,
    color: '#FFFFFF',
    fontSize: 13,
    borderWidth: 1,
    borderColor: '#244537',
  },
  sendBtn: {
    width: 44,
    height: 44,
    borderRadius: 10,
    backgroundColor: '#B7FF56',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

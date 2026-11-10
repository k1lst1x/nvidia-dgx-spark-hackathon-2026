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
  checkBridgeApiConnection,
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

function initialBridgeRoom() {
  if (typeof window !== 'undefined' && window.localStorage) {
    return window.localStorage.getItem('guidelens_bridge_room') || 'campus-bridge-session';
  }
  return 'campus-bridge-session';
}

function mergeBridgeMessages(current: BridgeMessage[], incoming: BridgeMessage[]) {
  const unique = new Map<string, BridgeMessage>();
  [...current, ...incoming].forEach((message) => {
    const key = `${message.session_id}|${message.sender_role}|${message.created_at}|${message.text}`;
    unique.set(key, message);
  });
  return [...unique.values()]
    .sort((left, right) => left.created_at.localeCompare(right.created_at))
    .slice(-50);
}

export default function DeafBlindBridgeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [sessionId, setSessionId] = useState(initialBridgeRoom);
  const [roomCode, setRoomCode] = useState(initialBridgeRoom);
  const [messages, setMessages] = useState<BridgeMessage[]>([]);
  const [customReply, setCustomReply] = useState('');
  const [customSignMessage, setCustomSignMessage] = useState('');
  const [bridgeOnline, setBridgeOnline] = useState<boolean | null>(null);
  const [bridgeNotice, setBridgeNotice] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastDeafSign, setLastDeafSign] = useState<string>('');
  const [lastBlindReply, setLastBlindReply] = useState<string>('');
  const [cameraFacing, setCameraFacing] = useState<'front' | 'back'>('front');
  const [isListeningMic, setIsListeningMic] = useState(false);

  const lastSpokenText = useRef<string>('');
  const scrollViewRef = useRef<any>(null);

  // Keep paired devices in sync, with local history as an offline fallback.
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const [msgs, online] = await Promise.all([
        getBridgeMessages(sessionId),
        checkBridgeApiConnection(),
      ]);
      if (!active) return;
      setBridgeOnline(online);
      setMessages((current) => mergeBridgeMessages(current, msgs));
      const latestDeaf = [...msgs].reverse().find((message) => message.sender_role === 'deaf');
      const latestBlind = [...msgs].reverse().find((message) => message.sender_role === 'blind');
      if (latestDeaf) setLastDeafSign(latestDeaf.text);
      if (latestBlind) setLastBlindReply(latestBlind.text);
    };
    void refresh();
    const interval = setInterval(() => void refresh(), 1800);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [sessionId]);

  const joinRoom = () => {
    const nextRoom = roomCode.trim().replace(/\s+/g, '-').slice(0, 64);
    if (!nextRoom) {
      setBridgeNotice('Enter a room code to join the conversation.');
      return;
    }
    setSessionId(nextRoom);
    setRoomCode(nextRoom);
    setMessages([]);
    setLastDeafSign('');
    setLastBlindReply('');
    setBridgeNotice(`Joined room ${nextRoom}. Enter the same code on the other device.`);
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem('guidelens_bridge_room', nextRoom);
    }
  };

  // Request camera permission if not granted
  useEffect(() => {
    if (permission && !permission.granted && permission.canAskAgain) {
      void requestPermission();
    }
  }, [permission, requestPermission]);

  // Handle hand landmark sign context changes from HandTracker
  const handleContextChange = (state: SignContextState) => {
    // Send each confident recognized word once; never turn isolated letters
    // into a misleading message or repeat a held pose every few seconds.
    const detectedWord = state.wordState?.currentWord;
    if (!detectedWord) {
      lastSpokenText.current = '';
      return;
    }

    if (
      detectedWord !== lastSpokenText.current &&
      (state.wordState?.currentConfidence ?? 0) >= 0.80
    ) {
      lastSpokenText.current = detectedWord;
      void handleDeafSigned(detectedWord.toUpperCase());
    }
  };

  const handleDeafSigned = async (text: string) => {
    if (!text.trim()) return;
    setLastDeafSign(text);
    setBridgeNotice('');
    setIsSpeaking(true);

    // 1. Speak aloud for the blind student
    Speech.speak(text, {
      rate: 0.95,
      pitch: 1.0,
      onDone: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });

    // 2. Post to bridge message store
    try {
      const msg = await sendBridgeMessage(sessionId, 'deaf', text);
      setMessages((prev) => mergeBridgeMessages(prev, [msg]));
      setBridgeNotice(msg.delivery_status === 'sent' ? 'Message sent to the shared room.' : 'Saved on this device only. Reconnect the API to share it with the other device.');
      scrollViewRef.current?.scrollToEnd({ animated: true });
    } catch (error) {
      setBridgeNotice(error instanceof Error ? error.message : 'Could not save this message.');
    }
  };

  const handleBlindSpokenReply = async (text: string) => {
    if (!text.trim()) return;
    setLastBlindReply(text);
    setCustomReply('');
    setBridgeNotice('');

    // 1. Speak confirmation chime or feedback
    Speech.speak(`Displaying to deaf partner: ${text}`, {
      rate: 1.1,
    });

    // 2. Post to bridge message store so deaf student sees it
    try {
      const msg = await sendBridgeMessage(sessionId, 'blind', text);
      setMessages((prev) => mergeBridgeMessages(prev, [msg]));
      setBridgeNotice(msg.delivery_status === 'sent' ? 'Reply sent to the shared room.' : 'Saved on this device only. Reconnect the API to share it with the other device.');
      scrollViewRef.current?.scrollToEnd({ animated: true });
    } catch (error) {
      setBridgeNotice(error instanceof Error ? error.message : 'Could not save this reply.');
    }
  };

  const handleSendTypedSign = () => {
    const text = customSignMessage.trim();
    if (!text) return;
    setCustomSignMessage('');
    void handleDeafSigned(text);
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
          onPress={() => (router.canGoBack() ? router.back() : router.replace('/hub'))}
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
          <Text style={s.subHeader}>Shared room · updates automatically</Text>
        </View>

        <View style={[s.localBadge, bridgeOnline === false && s.localBadgeOffline]}>
          <View style={[s.localDot, bridgeOnline === false && s.localDotOffline]} />
          <Text style={[s.localText, bridgeOnline === false && s.localTextOffline]}>{bridgeOnline === null ? 'CONNECTING' : bridgeOnline ? 'ROOM LIVE' : 'LOCAL ONLY'}</Text>
        </View>
      </View>

      <ScrollView
        ref={scrollViewRef}
        style={s.scroll}
        contentContainerStyle={s.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={s.roomCard}>
          <View style={s.roomCopy}>
            <Text style={s.roomTitle}>Conversation room</Text>
            <Text style={s.roomHint}>Use the same code on both devices.</Text>
          </View>
          <TextInput
            style={s.roomInput}
            value={roomCode}
            onChangeText={setRoomCode}
            placeholder="Enter room code"
            placeholderTextColor="#607B6E"
            autoCapitalize="none"
            accessibilityLabel="Conversation room code"
            onSubmitEditing={joinRoom}
          />
          <Pressable style={s.joinBtn} onPress={joinRoom} accessibilityRole="button">
            <Text style={s.joinBtnText}>Join</Text>
          </Pressable>
        </View>

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
                Sign, type a message, or choose a quick phrase:
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
          <View style={s.inputRow}>
            <TextInput
              style={s.customInput}
              value={customSignMessage}
              onChangeText={setCustomSignMessage}
              placeholder="Type a message for your partner…"
              placeholderTextColor="#607B6E"
              accessibilityLabel="Type a message as the deaf participant"
              onSubmitEditing={handleSendTypedSign}
              returnKeyType="send"
            />
            <Pressable style={s.sendBtn} onPress={handleSendTypedSign} accessibilityRole="button" accessibilityLabel="Send message">
              <Ionicons name="send" size={16} color="#07110F" />
            </Pressable>
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

        <View style={s.conversationCard} accessibilityLiveRegion="polite">
          <View style={s.conversationHeader}>
            <Text style={s.conversationTitle}>Conversation</Text>
            <Text style={s.conversationCount}>{messages.length} messages</Text>
          </View>
          {messages.length ? messages.slice(-12).map((message) => (
            <View key={`${message.session_id}-${message.sender_role}-${message.created_at}-${message.text}`} style={[s.messageBubble, message.sender_role === 'deaf' ? s.deafBubble : s.blindBubble]}>
              <View style={s.messageMeta}>
                <Text style={[s.messageRole, message.sender_role === 'blind' && s.messageRoleBlind]}>{message.sender_role === 'deaf' ? 'SIGN / TEXT' : 'VOICE / TEXT'}</Text>
                <Text style={s.messageTime}>{message.relative_time || 'just now'}</Text>
              </View>
              <Text style={s.messageText}>{message.text}</Text>
            </View>
          )) : (
            <Text style={s.emptyConversation}>Messages from either device will appear here.</Text>
          )}
          {bridgeNotice ? <Text style={s.bridgeNotice}>{bridgeNotice}</Text> : null}
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
            {Platform.OS === 'web' && <Pressable
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
            </Pressable>}
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
  roomCard: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#0B1A16',
    borderWidth: 1,
    borderColor: '#1C362C',
    borderRadius: 14,
    padding: 12,
    marginBottom: 14,
  },
  roomCopy: { flexGrow: 1, flexBasis: 140 },
  roomTitle: { color: '#F2F7F4', fontSize: 13, fontWeight: '800' },
  roomHint: { color: '#8FA99B', fontSize: 10, marginTop: 3 },
  roomInput: {
    flexGrow: 1,
    flexBasis: 150,
    height: 40,
    backgroundColor: '#11221D',
    borderRadius: 9,
    paddingHorizontal: 10,
    color: '#FFFFFF',
    fontSize: 12,
    borderWidth: 1,
    borderColor: '#244537',
  },
  joinBtn: { height: 40, paddingHorizontal: 16, justifyContent: 'center', borderRadius: 9, backgroundColor: '#B7FF56' },
  joinBtnText: { color: '#07110F', fontSize: 12, fontWeight: '900' },
  localBadgeOffline: { backgroundColor: '#302019' },
  localDotOffline: { backgroundColor: '#FF9A84' },
  localTextOffline: { color: '#FFB4A2' },
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
  conversationCard: {
    backgroundColor: '#091510',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#284437',
    padding: 12,
    gap: 8,
  },
  conversationHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 },
  conversationTitle: { color: '#F3F8F5', fontSize: 15, fontWeight: '900' },
  conversationCount: { color: '#91AA9D', fontSize: 10, fontWeight: '700' },
  messageBubble: { borderRadius: 10, padding: 10, borderLeftWidth: 3 },
  deafBubble: { backgroundColor: '#10251C', borderLeftColor: '#B7FF56' },
  blindBubble: { backgroundColor: '#241F0E', borderLeftColor: '#FFD043' },
  messageMeta: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  messageRole: { color: '#B7FF56', fontSize: 9, fontWeight: '900', letterSpacing: 0.6 },
  messageRoleBlind: { color: '#FFD043' },
  messageTime: { color: '#81998C', fontSize: 9 },
  messageText: { color: '#F5F8F6', fontSize: 14, lineHeight: 20, fontWeight: '600' },
  emptyConversation: { color: '#91AA9D', fontSize: 12, paddingVertical: 8 },
  bridgeNotice: { color: '#C8FF72', fontSize: 11, lineHeight: 16 },
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

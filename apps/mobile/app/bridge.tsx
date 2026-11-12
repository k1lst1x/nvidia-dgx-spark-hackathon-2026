import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router, useLocalSearchParams } from 'expo-router';
import * as Speech from 'expo-speech';
import { ExpoSpeechRecognitionModule, useSpeechRecognitionEvent } from 'expo-speech-recognition';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
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
import SignPlayer from '../components/SignPlayer';
import { translateEnglishToASLGloss } from '../src/aslGlossTranslator';
import {
  BridgeGoneError,
  LOCAL_BRIDGE_CODE,
  bridgeServerConfigured,
  createBridgeConversation,
  fetchBridgeMessages,
  flushBridgeOutbox,
  getBridgeConversation,
  joinBridgeConversation,
  loadLocalBridgeMessages,
  mergeBridgeMessages,
  sendBridgeMessage,
  type BridgeMessage,
  type BridgeRole,
} from '../src/campusMemory';
import { initialBridgeSignGate, updateBridgeSignGate } from '../src/bridgeSignGate';
import { signLabelToPhrase } from '../src/signPhrases';
import { useVisionLoop } from '../src/vision';

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

/** Which side of the conversation this device serves. */
type DeviceRole = BridgeRole | 'both';
type SyncState = 'local' | 'connecting' | 'synced' | 'reconnecting' | 'ended';

const POLL_MS = 1500;
const STATUS_EVERY_POLLS = 4;
// A sign is sent at this confidence, and only once until the hands drop or
// the word changes, so a held sign is not repeated every few seconds.

const ROLE_OPTIONS: Array<{ role: DeviceRole; title: string; detail: string; icon: keyof typeof Ionicons.glyphMap }> = [
  { role: 'deaf', title: 'I sign', detail: 'Deaf side: camera reads my signs, replies appear in large text.', icon: 'hand-right' },
  { role: 'blind', title: 'I speak', detail: 'Blind side: signs are spoken aloud, I reply by voice or buttons.', icon: 'mic' },
  { role: 'both', title: 'Both of us, this device', detail: 'One laptop shared by both people.', icon: 'people' },
];

function parseRole(value: unknown): DeviceRole | null {
  return value === 'deaf' || value === 'blind' || value === 'both' ? value : null;
}

function isBridgeCode(value: unknown): value is string {
  return typeof value === 'string' && /^\d{6}$/.test(value);
}

function messageKey(message: BridgeMessage) {
  return message.client_id || String(message.id);
}

function lastServerId(messages: BridgeMessage[]) {
  return messages.reduce((max, item) => (!item.pending && item.id > max ? item.id : max), 0);
}

function speakable(code: string) {
  return code.split('').join(' ');
}

export default function DeafBlindBridgeScreen() {
  const params = useLocalSearchParams<{ code?: string; role?: string }>();
  const [permission, requestPermission] = useCameraPermissions();
  const [role, setRole] = useState<DeviceRole | null>(parseRole(params.role));
  const [code, setCode] = useState<string | null>(null);
  const [joinInput, setJoinInput] = useState(isBridgeCode(params.code) ? params.code : '');
  const [setupError, setSetupError] = useState('');
  const [busy, setBusy] = useState(false);
  const [sync, setSync] = useState<SyncState>('local');
  const [partnerJoined, setPartnerJoined] = useState(false);
  const [messages, setMessages] = useState<BridgeMessage[]>([]);
  const [customReply, setCustomReply] = useState('');
  const [customSignMessage, setCustomSignMessage] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [spelled, setSpelled] = useState('');
  const [cameraFacing, setCameraFacing] = useState<'front' | 'back'>('front');
  const [isListeningMic, setIsListeningMic] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const nativeDeafRole = Platform.OS !== 'web' && (role === 'deaf' || role === 'both');
  const { result: nativeSignResult, state: nativeSignState } = useVisionLoop(
    cameraRef,
    'sign',
    Boolean(code && nativeDeafRole && permission?.granted && bridgeServerConfigured()),
  );

  const roleRef = useRef(role);
  const codeRef = useRef(code);
  const messagesRef = useRef(messages);
  roleRef.current = role;
  codeRef.current = code;
  messagesRef.current = messages;
  const partnerRef = useRef(partnerJoined);
  partnerRef.current = partnerJoined;
  const handled = useRef<Set<string>>(new Set());
  // Partner state already announced, so joining an occupied room is not said twice.
  const previousPartner = useRef(false);
  const signGate = useRef(initialBridgeSignGate());
  const autoJoinTried = useRef(false);
  const aslEngineRef = useRef<{ backspace: () => string; addSpace: () => string; clearText: () => void } | null>(null);
  const scrollViewRef = useRef<{ scrollToEnd: (options?: { animated?: boolean }) => void } | null>(null);

  const speak = useCallback((text: string, rate = 0.95) => {
    setIsSpeaking(true);
    Speech.speak(text, {
      rate,
      onDone: () => setIsSpeaking(false),
      onStopped: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  }, []);

  /** Add messages to the conversation; announce the partner's new ones. */
  const receive = useCallback((incoming: BridgeMessage[], announce: boolean) => {
    if (!incoming.length) return;
    setMessages((previous) => mergeBridgeMessages(previous, incoming));
    incoming.forEach((message) => {
      const key = messageKey(message);
      if (handled.current.has(key)) return;
      handled.current.add(key);
      if (!announce || message.from_this_device) return;
      if (message.sender_role === 'deaf' && roleRef.current === 'blind') {
        speak(message.text);
      }
      if (message.sender_role === 'blind' && roleRef.current === 'deaf' && typeof navigator !== 'undefined') {
        navigator.vibrate?.(150);
      }
    });
    setTimeout(() => scrollViewRef.current?.scrollToEnd({ animated: true }), 50);
  }, [speak]);

  useEffect(() => {
    if (permission && !permission.granted && permission.canAskAgain && Platform.OS !== 'web') {
      void requestPermission();
    }
  }, [permission, requestPermission]);

  const enter = useCallback((nextCode: string, nextRole: DeviceRole, joined: boolean) => {
    const local = loadLocalBridgeMessages(nextCode);
    handled.current = new Set(local.map(messageKey));
    setMessages(local);
    setPartnerJoined(joined);
    previousPartner.current = joined;
    setRole(nextRole);
    setCode(nextCode);
    setSync(nextCode === LOCAL_BRIDGE_CODE ? 'local' : 'connecting');
    setSetupError('');
    signGate.current = initialBridgeSignGate();
    if (nextCode !== LOCAL_BRIDGE_CODE) router.setParams({ code: nextCode, role: nextRole });
  }, []);

  const start = async (chosen: DeviceRole) => {
    if (!bridgeServerConfigured()) {
      setSetupError('No GuideLens server is set, so two devices cannot connect. Use "This device only", or set the server address in the Hub.');
      return;
    }
    setBusy(true);
    try {
      const conversation = await createBridgeConversation();
      enter(conversation.code, chosen, conversation.partner_joined);
      speak(`Your bridge code is ${speakable(conversation.code)}. Share it with your partner.`, 0.9);
    } catch {
      setSetupError('Could not reach the GuideLens server. Check the connection, or use "This device only".');
    } finally {
      setBusy(false);
    }
  };

  const join = useCallback(async (chosen: DeviceRole, joinCode: string) => {
    if (!isBridgeCode(joinCode)) {
      setSetupError('A bridge code is six digits.');
      return;
    }
    if (!bridgeServerConfigured()) {
      setSetupError('No GuideLens server is set, so this device cannot join a partner.');
      return;
    }
    setBusy(true);
    try {
      const conversation = await joinBridgeConversation(joinCode);
      enter(conversation.code, chosen, conversation.partner_joined);
      if (chosen !== 'deaf') speak('Connected to your partner.');
    } catch (error) {
      setSetupError(error instanceof BridgeGoneError ? error.message : 'Could not reach the GuideLens server.');
    } finally {
      setBusy(false);
    }
  }, [enter, speak]);

  // A shared link (/bridge?code=123456&role=blind) joins straight away.
  useEffect(() => {
    if (autoJoinTried.current || code) return;
    autoJoinTried.current = true;
    const linkRole = parseRole(params.role);
    if (linkRole && isBridgeCode(params.code)) void join(linkRole, params.code);
  }, [code, join, params.code, params.role]);

  const leave = () => {
    Speech.stop();
    setCode(null);
    setMessages([]);
    setSync('local');
    setPartnerJoined(false);
    setJoinInput('');
    signGate.current = initialBridgeSignGate();
    router.setParams({ code: undefined, role: undefined } as never);
  };

  // Live refresh: send anything queued, then fetch only what is new.
  useEffect(() => {
    if (!code || code === LOCAL_BRIDGE_CODE) return undefined;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let polls = 0;
    let firstFetch = true;
    const tick = async () => {
      try {
        receive(await flushBridgeOutbox(code), false);
        const after = firstFetch ? undefined : lastServerId(messagesRef.current) || undefined;
        const incoming = await fetchBridgeMessages(code, after);
        if (cancelled) return;
        // History that was already there when this device opened is not re-announced.
        receive(incoming, !firstFetch);
        firstFetch = false;
        // Check every refresh until the partner arrives, then only now and then.
        if (!partnerRef.current || polls % STATUS_EVERY_POLLS === 0) {
          const status = await getBridgeConversation(code);
          if (!cancelled) setPartnerJoined(status.partner_joined);
        }
        polls += 1;
        if (!cancelled) setSync('synced');
      } catch (error) {
        if (cancelled) return;
        if (error instanceof BridgeGoneError) {
          setSync('ended');
          return;
        }
        setSync('reconnecting');
      }
      if (!cancelled) timer = setTimeout(() => void tick(), POLL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [code, receive]);

  useEffect(() => {
    if (partnerJoined && !previousPartner.current && role === 'blind') speak('Your partner has joined.');
    previousPartner.current = partnerJoined;
  }, [partnerJoined, role, speak]);

  const send = async (senderRole: BridgeRole, text: string) => {
    const activeCode = codeRef.current;
    const cleaned = text.trim();
    if (!activeCode || !cleaned) return;
    try {
      const message = await sendBridgeMessage(activeCode, senderRole, cleaned);
      receive([message], false);
    } catch (error) {
      if (error instanceof BridgeGoneError) setSync('ended');
    }
  };

  const handleDeafSigned = (text: string) => {
    const phrase = text.trim();
    if (!phrase) return;
    // The listener is at this device only when both people share it.
    if (roleRef.current === 'both') speak(phrase);
    void send('deaf', phrase);
  };

  // Native phones use the configured API's phrase decoder. The browser keeps
  // its on-device word engine, and both paths retain quick-sign fallback.
  useEffect(() => {
    const phrase = nativeSignResult?.should_speak ? nativeSignResult.speech.trim() : '';
    if (!phrase) return;
    handleDeafSigned(phrase);
  }, [nativeSignResult?.scene_key, nativeSignResult?.should_speak, nativeSignResult?.speech]);

  const handleBlindReply = (text: string) => {
    if (!text.trim()) return;
    setCustomReply('');
    if (roleRef.current === 'blind') speak('Sent.', 1.1);
    void send('blind', text);
  };

  useSpeechRecognitionEvent('start', () => setIsListeningMic(true));
  useSpeechRecognitionEvent('end', () => setIsListeningMic(false));
  useSpeechRecognitionEvent('error', (event) => {
    setIsListeningMic(false);
    if (event.error !== 'aborted' && event.error !== 'no-speech') {
      setSetupError('Voice reply could not start. Use a quick reply or type your message.');
    }
  });
  useSpeechRecognitionEvent('result', (event) => {
    if (!event.isFinal) return;
    const transcript = event.results[0]?.transcript.trim();
    if (transcript) handleBlindReply(transcript);
  });

  const handleContextChange = (state: SignContextState) => {
    setSpelled(state.aslState?.spelledText?.trim() ?? '');
    const word = state.wordState?.currentWord ?? '';
    const confidence = state.wordState?.currentConfidence ?? 0;
    const next = updateBridgeSignGate(signGate.current, state.handCount, word, confidence);
    signGate.current = next.gate;
    if (next.shouldSend) handleDeafSigned(signLabelToPhrase(word));
  };

  const sendTypedSign = () => {
    const text = customSignMessage.trim();
    if (!text) return;
    setCustomSignMessage('');
    handleDeafSigned(text);
  };

  const sendSpelledWord = () => {
    if (!spelled) return;
    handleDeafSigned(spelled.toUpperCase());
    aslEngineRef.current?.clearText();
    setSpelled('');
  };

  const toggleMicListening = async () => {
    if (isListeningMic) {
      ExpoSpeechRecognitionModule.stop();
      setIsListeningMic(false);
      return;
    }
    try {
      if (!ExpoSpeechRecognitionModule.isRecognitionAvailable()) {
        setSetupError('Speech recognition is not available on this device. Use a quick reply or type your message.');
        return;
      }
      const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!permission.granted) {
        setSetupError('Microphone and speech-recognition permission are needed for voice replies.');
        return;
      }
      ExpoSpeechRecognitionModule.start({ lang: 'en-US', interimResults: false, continuous: false });
    } catch {
      setIsListeningMic(false);
      setSetupError('Voice reply could not start. Use a quick reply or type your message.');
    }
  };

  const lastDeaf = [...messages].reverse().find((item) => item.sender_role === 'deaf');
  const lastBlind = [...messages].reverse().find((item) => item.sender_role === 'blind');
  const lastBlindAsl = useMemo(() => {
    if (!lastBlind?.text) return null;
    return translateEnglishToASLGloss(lastBlind.text);
  }, [lastBlind?.text]);
  const pendingCount = messages.filter((item) => item.pending).length;
  const shareLink = code && code !== LOCAL_BRIDGE_CODE && Platform.OS === 'web' && typeof window !== 'undefined'
    ? `${window.location.origin}/bridge?code=${code}&role=${role === 'deaf' ? 'blind' : 'deaf'}`
    : '';

  const badge = (() => {
    if (!code) return { text: 'NOT CONNECTED', color: '#9EB5AA' };
    if (code === LOCAL_BRIDGE_CODE) return { text: 'THIS DEVICE ONLY', color: '#9EB5AA' };
    if (sync === 'synced') return { text: partnerJoined ? 'PARTNER CONNECTED' : 'WAITING FOR PARTNER', color: '#B7FF56' };
    if (sync === 'ended') return { text: 'CONVERSATION ENDED', color: '#FF8A7A' };
    if (sync === 'reconnecting') return { text: 'RECONNECTING…', color: '#FFD043' };
    return { text: 'CONNECTING…', color: '#FFD043' };
  })();

  const header = (
    <View style={s.header}>
      <Pressable
        style={s.backBtn}
        onPress={() => (code ? leave() : router.canGoBack() ? router.back() : router.replace('/hub'))}
        accessibilityRole="button"
        accessibilityLabel={code ? 'Leave this conversation' : 'Back to Assistant Hub'}
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
      <View style={s.localBadge} accessibilityLabel={`Status: ${badge.text}`}>
        <View style={[s.localDot, { backgroundColor: badge.color }]} />
        <Text style={[s.localText, { color: badge.color }]}>{badge.text}</Text>
      </View>
    </View>
  );

  if (!code) {
    return (
      <SafeAreaView style={s.safe}>
        {header}
        <ScrollView style={s.scroll} contentContainerStyle={s.scrollContent} keyboardShouldPersistTaps="handled">
          <Text style={s.setupTitle}>Who is using this device?</Text>
          <View style={s.roleList}>
            {ROLE_OPTIONS.map((option) => (
              <Pressable
                key={option.role}
                style={[s.roleCard, role === option.role && s.roleCardActive]}
                onPress={() => setRole(option.role)}
                accessibilityRole="radio"
                accessibilityState={{ selected: role === option.role }}
                accessibilityLabel={`${option.title}. ${option.detail}`}
              >
                <Ionicons name={option.icon} size={22} color={role === option.role ? '#07110F' : '#B7FF56'} />
                <View style={s.roleCardBody}>
                  <Text style={[s.roleCardTitle, role === option.role && s.roleCardTitleActive]}>{option.title}</Text>
                  <Text style={[s.roleCardDetail, role === option.role && s.roleCardTitleActive]}>{option.detail}</Text>
                </View>
              </Pressable>
            ))}
          </View>

          {role && (
            <View style={s.setupActions}>
              <Pressable
                style={[s.primaryBtn, busy && s.btnDisabled]}
                disabled={busy}
                onPress={() => void start(role)}
                accessibilityRole="button"
                accessibilityLabel="Start a new conversation and get a code"
              >
                <Text style={s.primaryBtnText}>Start a conversation</Text>
              </Pressable>

              <Text style={s.setupLabel}>Or join your partner with their 6-digit code</Text>
              <View style={s.inputRow}>
                <TextInput
                  style={[s.customInput, s.codeInput]}
                  value={joinInput}
                  onChangeText={(value) => setJoinInput(value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="123456"
                  placeholderTextColor="#607B6E"
                  keyboardType="number-pad"
                  maxLength={6}
                  accessibilityLabel="Bridge code, six digits"
                  onSubmitEditing={() => void join(role, joinInput)}
                />
                <Pressable
                  style={[s.joinBtn, busy && s.btnDisabled]}
                  disabled={busy}
                  onPress={() => void join(role, joinInput)}
                  accessibilityRole="button"
                  accessibilityLabel="Join with this code"
                >
                  <Text style={s.joinBtnText}>Join</Text>
                </Pressable>
              </View>

              <Pressable
                style={s.secondaryBtn}
                onPress={() => enter(LOCAL_BRIDGE_CODE, 'both', false)}
                accessibilityRole="button"
                accessibilityLabel="Use this device only, without a partner device"
              >
                <Text style={s.secondaryBtnText}>This device only (no second device)</Text>
              </Pressable>
              {busy && <ActivityIndicator color="#B7FF56" />}
              {setupError ? <Text style={s.errorText} accessibilityLiveRegion="polite">{setupError}</Text> : null}
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    );
  }

  const showDeaf = role === 'deaf' || role === 'both';
  const showBlind = role === 'blind' || role === 'both';

  return (
    <SafeAreaView style={s.safe}>
      {header}

      <ScrollView
        ref={scrollViewRef as never}
        style={s.scroll}
        contentContainerStyle={s.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {code !== LOCAL_BRIDGE_CODE && (
          <View style={s.codeBar}>
            <View style={s.codeBarBody}>
              <Text style={s.codeLabel}>CONVERSATION CODE</Text>
              <Text style={s.codeValue} accessibilityLabel={`Conversation code ${speakable(code)}`}>{code}</Text>
              {shareLink ? <Text style={s.shareLink} selectable>{shareLink}</Text> : null}
            </View>
            <Pressable style={s.leaveBtn} onPress={leave} accessibilityRole="button" accessibilityLabel="Leave conversation">
              <Text style={s.leaveBtnText}>Leave</Text>
            </Pressable>
          </View>
        )}
        {sync === 'ended' && (
          <Text style={s.errorText} accessibilityLiveRegion="assertive">
            This conversation has ended or expired. Leave and start a new one.
          </Text>
        )}
        {pendingCount > 0 && sync !== 'ended' && (
          <Text style={s.pendingNote}>{pendingCount} message{pendingCount === 1 ? '' : 's'} waiting to send…</Text>
        )}

        {showDeaf && (
          <View style={s.panelDeaf}>
            <View style={s.panelHeader}>
              <View style={s.roleTagDeaf}>
                <Ionicons name="hand-right" size={16} color="#B7FF56" />
                <Text style={s.roleTagTextDeaf}>DEAF STUDENT (SIGNS ASL)</Text>
              </View>
              {isSpeaking && role === 'both' && (
                <View style={s.speakingPill}>
                  <Ionicons name="volume-high" size={14} color="#07110F" />
                  <Text style={s.speakingPillText}>Speaking…</Text>
                </View>
              )}
            </View>

            <View style={s.cameraCard}>
              {Platform.OS === 'web' ? (
                <HandTracker onContextChange={handleContextChange} aslEngineRef={aslEngineRef} showLandmarks />
              ) : permission?.granted ? (
                <CameraView ref={cameraRef} facing={cameraFacing} style={StyleSheet.absoluteFill} />
              ) : (
                <View style={s.cameraFallback}>
                  <Ionicons name="videocam-off" size={32} color="#607B6E" />
                  <Text style={s.fallbackText}>Camera initializing…</Text>
                </View>
              )}
              <View style={s.cameraOverlayBar}>
                <Text style={s.overlayPrompt}>
                  {Platform.OS === 'web'
                    ? 'Sign with hands in frame or pick a sign:'
                    : !bridgeServerConfigured() || nativeSignState === 'offline'
                      ? 'Sign decoding needs the GuideLens server. Use a quick sign or type:'
                      : 'Reading your signed phrase… or use a quick sign:'}
                </Text>
                {Platform.OS !== 'web' && (
                  <Pressable style={s.flipBtn} onPress={() => setCameraFacing((c) => (c === 'front' ? 'back' : 'front'))}>
                    <Ionicons name="camera-reverse" size={16} color="#FFFFFF" />
                  </Pressable>
                )}
              </View>
            </View>

            <View style={s.signedCard}>
              <Text style={s.signedLabel}>{role === 'both' ? 'SPOKEN TO BLIND PARTNER:' : 'SENT TO PARTNER:'}</Text>
              <Text style={s.signedText}>
                {lastDeaf ? `"${lastDeaf.text}"${lastDeaf.pending ? ' · sending…' : ''}` : 'Waiting for a sign…'}
              </Text>
            </View>

            {spelled ? (
              <View style={s.spelledRow}>
                <Text style={s.spelledText}>Spelled: {spelled.toUpperCase()}</Text>
                <Pressable style={s.spelledSend} onPress={sendSpelledWord} accessibilityRole="button" accessibilityLabel="Send the spelled word">
                  <Text style={s.spelledSendText}>Send</Text>
                </Pressable>
                <Pressable
                  style={s.spelledClear}
                  onPress={() => { aslEngineRef.current?.clearText(); setSpelled(''); }}
                  accessibilityRole="button"
                  accessibilityLabel="Clear the spelled letters"
                >
                  <Text style={s.spelledClearText}>Clear</Text>
                </Pressable>
              </View>
            ) : null}

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
                placeholder="Or type a message for your partner…"
                placeholderTextColor="#607B6E"
                accessibilityLabel="Type a message as the deaf participant"
                onSubmitEditing={sendTypedSign}
                returnKeyType="send"
              />
              <Pressable style={s.sendBtn} onPress={sendTypedSign} accessibilityRole="button" accessibilityLabel="Send message">
                <Ionicons name="send" size={16} color="#07110F" />
              </Pressable>
            </View>

            {role === 'deaf' && (
              <View style={s.giantTextCard}>
                <Text style={s.giantTextLabel}>YOUR PARTNER SAID:</Text>
                <Text style={s.giantText} accessibilityLiveRegion="assertive">
                  {lastBlind ? lastBlind.text.toUpperCase() : 'WAITING FOR A REPLY'}
                </Text>
                {lastBlindAsl && lastBlindAsl.tokens.length > 0 && (
                  <View style={s.bridgeSignPlayerBox}>
                    <View style={s.bridgeSignPlayerTitleRow}>
                      <Ionicons name="hand-left" size={13} color="#C8FF72" />
                      <Text style={s.bridgeSignPlayerTitle}>
                        ASL SIGN PLAYBACK: {lastBlindAsl.aslSentence}
                      </Text>
                    </View>
                    <SignPlayer tokens={lastBlindAsl.tokens} height={240} autoPlay={true} />
                  </View>
                )}
              </View>
            )}
          </View>
        )}

        {role === 'both' && (
          <View style={s.dividerRow}>
            <View style={s.dividerLine} />
            <View style={s.dividerBadge}>
              <Ionicons name="swap-vertical" size={16} color="#B7FF56" />
            </View>
            <View style={s.dividerLine} />
          </View>
        )}

        {showBlind && (
          <View style={s.panelBlind}>
            <View style={s.panelHeader}>
              <View style={s.roleTagBlind}>
                <Ionicons name="mic" size={16} color="#FFD043" />
                <Text style={s.roleTagTextBlind}>BLIND STUDENT (VOICE REPLY)</Text>
              </View>
              <Text style={s.displayHint}>{role === 'both' ? 'Displays large text for deaf partner' : 'Replies appear on your partner’s screen'}</Text>
            </View>

            {role === 'blind' && (
              <Pressable
                style={s.heardCard}
                onPress={() => lastDeaf && speak(lastDeaf.text)}
                accessibilityRole="button"
                accessibilityLabel={lastDeaf ? `Partner signed: ${lastDeaf.text}. Tap to hear it again.` : 'No sign yet'}
              >
                <Text style={s.giantTextLabel}>YOUR PARTNER SIGNED (TAP TO REPEAT):</Text>
                <Text style={s.heardText}>{lastDeaf ? lastDeaf.text : 'Waiting for your partner…'}</Text>
              </Pressable>
            )}

            {role === 'both' && (
              <View style={s.giantTextCard}>
                <Text style={s.giantTextLabel}>SHOWN TO DEAF PARTNER:</Text>
                <Text style={s.giantText} accessibilityRole="text" accessibilityLiveRegion="assertive">
                  {lastBlind ? lastBlind.text.toUpperCase() : 'PRESS A REPLY BELOW OR SPEAK'}
                </Text>
                {lastBlindAsl && lastBlindAsl.tokens.length > 0 && (
                  <View style={s.bridgeSignPlayerBox}>
                    <View style={s.bridgeSignPlayerTitleRow}>
                      <Ionicons name="hand-left" size={13} color="#C8FF72" />
                      <Text style={s.bridgeSignPlayerTitle}>
                        ASL SIGN PLAYBACK: {lastBlindAsl.aslSentence}
                      </Text>
                    </View>
                    <SignPlayer tokens={lastBlindAsl.tokens} height={240} autoPlay={true} />
                  </View>
                )}
              </View>
            )}

            <Text style={s.sectionSubtitle}>Quick replies (touch to send):</Text>
            <View style={s.repliesGrid}>
              {QUICK_BLIND_REPLIES.map((reply) => (
                <Pressable
                  key={reply}
                  style={s.replyCard}
                  onPress={() => handleBlindReply(reply)}
                  accessibilityRole="button"
                  accessibilityLabel={reply}
                >
                  <Ionicons name="volume-medium-outline" size={16} color="#FFD043" />
                  <Text style={s.replyCardText}>{reply}</Text>
                </Pressable>
              ))}
            </View>

            <View style={s.inputRow}>
              <Pressable
                style={[s.micBtn, isListeningMic && s.micBtnActive]}
                onPress={() => void toggleMicListening()}
                accessibilityRole="button"
                accessibilityLabel={isListeningMic ? 'Stop microphone' : 'Speak reply'}
              >
                <Ionicons name={isListeningMic ? 'mic' : 'mic-outline'} size={22} color={isListeningMic ? '#07110F' : '#FFD043'} />
              </Pressable>
              <TextInput
                style={s.customInput}
                value={customReply}
                onChangeText={setCustomReply}
                placeholder="Or type a custom reply…"
                placeholderTextColor="#607B6E"
                onSubmitEditing={() => handleBlindReply(customReply)}
              />
              <Pressable style={s.sendBtn} onPress={() => handleBlindReply(customReply)} accessibilityRole="button" accessibilityLabel="Send reply">
                <Ionicons name="send" size={16} color="#07110F" />
              </Pressable>
            </View>
          </View>
        )}

        <View style={s.transcript}>
          <Text style={s.transcriptTitle}>CONVERSATION</Text>
          {messages.length === 0 ? (
            <Text style={s.transcriptEmpty}>No messages yet.</Text>
          ) : (
            messages.map((message) => (
              <View
                key={messageKey(message)}
                style={[s.bubble, message.sender_role === 'deaf' ? s.bubbleDeaf : s.bubbleBlind]}
                accessibilityLabel={`${message.sender_role === 'deaf' ? 'Signed' : 'Said'}: ${message.text}`}
              >
                <Text style={s.bubbleMeta}>
                  {message.sender_role === 'deaf' ? 'SIGNED' : 'SAID'}
                  {message.pending ? ' · sending…' : ''}
                </Text>
                <Text style={s.bubbleText}>{message.text}</Text>
              </View>
            ))
          )}
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
  bridgeSignPlayerBox: {
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#1A2E24',
    gap: 8,
  },
  bridgeSignPlayerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  bridgeSignPlayerTitle: {
    color: '#C8FF72',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.6,
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
  setupTitle: { color: '#FFFFFF', fontSize: 18, fontWeight: '800', marginBottom: 12 },
  roleList: { gap: 10, marginBottom: 18 },
  roleCard: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#0B1A16', borderWidth: 1, borderColor: '#1C362C', borderRadius: 14, padding: 14 },
  roleCardActive: { backgroundColor: '#B7FF56', borderColor: '#B7FF56' },
  roleCardBody: { flex: 1 },
  roleCardTitle: { color: '#FFFFFF', fontSize: 15, fontWeight: '800' },
  roleCardTitleActive: { color: '#07110F' },
  roleCardDetail: { color: '#9EB5AA', fontSize: 12, marginTop: 2 },
  setupActions: { gap: 12 },
  setupLabel: { color: '#9EB5AA', fontSize: 12, fontWeight: '600', marginTop: 6 },
  primaryBtn: { backgroundColor: '#B7FF56', borderRadius: 12, paddingVertical: 14, alignItems: 'center' },
  primaryBtnText: { color: '#07110F', fontSize: 15, fontWeight: '900' },
  secondaryBtn: { borderWidth: 1, borderColor: '#2B4237', borderRadius: 12, paddingVertical: 12, alignItems: 'center' },
  secondaryBtnText: { color: '#D5E6D8', fontSize: 13, fontWeight: '700' },
  btnDisabled: { opacity: 0.5 },
  codeInput: { fontSize: 20, letterSpacing: 4, fontWeight: '800' },
  joinBtn: { height: 44, paddingHorizontal: 18, borderRadius: 10, backgroundColor: '#B7FF56', justifyContent: 'center' },
  joinBtnText: { color: '#07110F', fontWeight: '900', fontSize: 14 },
  errorText: { color: '#FF8A7A', fontSize: 13, fontWeight: '700', marginBottom: 10 },
  codeBar: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#0B1A16', borderWidth: 1, borderColor: '#1C362C', borderRadius: 14, padding: 12, marginBottom: 12 },
  codeBarBody: { flex: 1 },
  codeLabel: { color: '#769485', fontSize: 10, fontWeight: '800', letterSpacing: 0.5 },
  codeValue: { color: '#B7FF56', fontSize: 28, fontWeight: '900', letterSpacing: 6 },
  shareLink: { color: '#9EB5AA', fontSize: 11, marginTop: 4 },
  leaveBtn: { borderWidth: 1, borderColor: '#2B4237', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8 },
  leaveBtnText: { color: '#D5E6D8', fontWeight: '800', fontSize: 12 },
  pendingNote: { color: '#FFD043', fontSize: 12, fontWeight: '700', marginBottom: 10 },
  spelledRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 12 },
  spelledText: { color: '#FFFFFF', fontSize: 14, fontWeight: '800', flex: 1 },
  spelledSend: { backgroundColor: '#B7FF56', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6 },
  spelledSendText: { color: '#07110F', fontWeight: '900', fontSize: 12 },
  spelledClear: { borderWidth: 1, borderColor: '#2B4237', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 },
  spelledClearText: { color: '#D5E6D8', fontWeight: '800', fontSize: 12 },
  heardCard: { backgroundColor: '#000000', borderWidth: 2, borderColor: '#B7FF56', borderRadius: 12, padding: 16, marginBottom: 12 },
  heardText: { color: '#FFFFFF', fontSize: 30, fontWeight: '900', lineHeight: 36 },
  transcript: { marginTop: 18, gap: 8 },
  transcriptTitle: { color: '#769485', fontSize: 11, fontWeight: '800', letterSpacing: 0.5 },
  transcriptEmpty: { color: '#607B6E', fontSize: 12 },
  bubble: { borderRadius: 12, padding: 10, maxWidth: '85%' },
  bubbleDeaf: { alignSelf: 'flex-start', backgroundColor: '#132A20', borderWidth: 1, borderColor: '#244537' },
  bubbleBlind: { alignSelf: 'flex-end', backgroundColor: '#2A2410', borderWidth: 1, borderColor: '#4A3D14' },
  bubbleMeta: { color: '#9EB5AA', fontSize: 9, fontWeight: '800', letterSpacing: 0.5, marginBottom: 2 },
  bubbleText: { color: '#FFFFFF', fontSize: 14, fontWeight: '600' },
});

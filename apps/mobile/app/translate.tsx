import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { ExpoSpeechRecognitionModule, useSpeechRecognitionEvent } from 'expo-speech-recognition';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import SignPlayer from '../components/SignPlayer';
import {
  translateEnglishToASLGloss,
  translateEnglishToSign,
  type ASLTranslationResult,
} from '../src/aslGlossTranslator';
import { getApiUrl } from '../src/vision';

const PRESET_PHRASES = [
  'Where is Room 204?',
  'Please help me',
  'Thank you very much',
  'Follow me',
  'I need water',
  'Wait a moment',
  'Yes, I understand',
  'SFBU DGX Spark',
];

export default function TranslateScreen() {
  const [inputText, setInputText] = useState('Where is Room 204?');
  const [result, setResult] = useState<ASLTranslationResult>(() =>
    translateEnglishToASLGloss('Where is Room 204?'),
  );
  const [isTranslating, setIsTranslating] = useState(false);
  const [isListeningMic, setIsListeningMic] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>('Where is Room 204?');

  const apiUrl = useMemo(() => getApiUrl(), []);

  // Speech recognition events
  useSpeechRecognitionEvent('start', () => {
    setIsListeningMic(true);
    setMicError(null);
  });

  useSpeechRecognitionEvent('end', () => {
    setIsListeningMic(false);
  });

  useSpeechRecognitionEvent('error', (event) => {
    setIsListeningMic(false);
    if (event.error !== 'aborted' && event.error !== 'no-speech') {
      setMicError('Speech recognition unavailable. Type your phrase below.');
    }
  });

  useSpeechRecognitionEvent('result', (event) => {
    if (!event.isFinal) return;
    const transcript = event.results[0]?.transcript?.trim();
    if (transcript) {
      setInputText(transcript);
      setSelectedPreset('');
      handleTranslate(transcript);
    }
  });

  const toggleMic = async () => {
    if (isListeningMic) {
      try {
        ExpoSpeechRecognitionModule.stop();
      } catch {
        // ignore
      }
      setIsListeningMic(false);
      return;
    }

    setMicError(null);
    try {
      if (!ExpoSpeechRecognitionModule.isRecognitionAvailable()) {
        setMicError('Speech recognition is not available in this environment. Type English text.');
        return;
      }
      const perm = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!perm.granted) {
        setMicError('Microphone permission required for speech-to-sign translation.');
        return;
      }
      ExpoSpeechRecognitionModule.start({
        lang: 'en-US',
        interimResults: false,
        continuous: false,
      });
    } catch {
      setIsListeningMic(false);
      setMicError('Could not start speech recognition. You can type text directly.');
    }
  };

  const handleTranslate = useCallback(
    async (textToTranslate?: string) => {
      const q = (textToTranslate !== undefined ? textToTranslate : inputText).trim();
      if (!q) return;

      setIsTranslating(true);
      try {
        const res = await translateEnglishToSign(q, apiUrl);
        setResult(res);
      } finally {
        setIsTranslating(false);
      }
    },
    [inputText, apiUrl],
  );

  const handleSelectPreset = (phrase: string) => {
    setInputText(phrase);
    setSelectedPreset(phrase);
    handleTranslate(phrase);
  };

  return (
    <SafeAreaView style={s.safe}>
      <KeyboardAvoidingView
        style={s.container}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {/* Top App Bar */}
        <View style={s.header}>
          <Pressable
            onPress={() => router.back()}
            style={s.backBtn}
            accessibilityRole="button"
            accessibilityLabel="Go back"
          >
            <Ionicons name="arrow-back" size={20} color="#F1F5F3" />
          </Pressable>

          <View style={s.headerTitleWrap}>
            <View style={s.headerTitleRow}>
              <Ionicons name="swap-horizontal" size={17} color="#B7FF56" />
              <Text style={s.headerTitle}>ASL Sign Studio</Text>
            </View>
            <Text style={s.headerSubtitle}>English ⇄ American Sign Language</Text>
          </View>

          <View style={s.modeBadge}>
            <View style={s.modeDot} />
            <Text style={s.modeBadgeText}>Offline-Ready</Text>
          </View>
        </View>

        <ScrollView
          style={s.scroll}
          contentContainerStyle={s.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Direction Banner / Switcher */}
          <View style={s.directionCard}>
            <View style={s.directionPillActive}>
              <Ionicons name="chatbubble-ellipses-outline" size={14} color="#07110F" />
              <Text style={s.directionPillActiveText}>English (Voice / Text)</Text>
            </View>
            <View style={s.directionArrow}>
              <Ionicons name="arrow-forward" size={16} color="#B7FF56" />
            </View>
            <View style={s.directionPillTarget}>
              <Ionicons name="hand-left" size={14} color="#B7FF56" />
              <Text style={s.directionPillTargetText}>ASL Animated Signs</Text>
            </View>
            <Pressable
              onPress={() => router.push({ pathname: '/camera', params: { initialMode: 'sign' } })}
              style={s.switchModeBtn}
              accessibilityRole="button"
              accessibilityLabel="Switch to Camera Sign to Voice Interpreter"
            >
              <Ionicons name="camera-outline" size={14} color="#C8FF72" />
              <Text style={s.switchModeBtnText}>Sign Camera</Text>
            </Pressable>
          </View>

          {/* Text & Voice Input Box */}
          <View style={s.inputCard}>
            <View style={s.inputHeaderRow}>
              <Text style={s.inputHeaderLabel}>ENGLISH INPUT (VOICE OR TEXT)</Text>
              {inputText.length > 0 && (
                <Pressable
                  onPress={() => {
                    setInputText('');
                    setSelectedPreset('');
                  }}
                  accessibilityRole="button"
                  accessibilityLabel="Clear text"
                  hitSlop={8}
                >
                  <Text style={s.clearTextBtn}>Clear</Text>
                </Pressable>
              )}
            </View>

            <View style={s.textInputRow}>
              <TextInput
                style={s.textInput}
                value={inputText}
                onChangeText={(val) => {
                  setInputText(val);
                  setSelectedPreset('');
                }}
                onSubmitEditing={() => handleTranslate()}
                placeholder="Speak or type English phrase..."
                placeholderTextColor="#667C73"
                returnKeyType="go"
                multiline={false}
                accessibilityLabel="English phrase to translate to ASL"
              />

              <Pressable
                onPress={toggleMic}
                style={[s.micBtn, isListeningMic && s.micBtnActive]}
                accessibilityRole="button"
                accessibilityLabel={isListeningMic ? 'Stop listening' : 'Start voice input'}
              >
                <Ionicons
                  name={isListeningMic ? 'mic' : 'mic-outline'}
                  size={20}
                  color={isListeningMic ? '#07110F' : '#B7FF56'}
                />
              </Pressable>

              <Pressable
                onPress={() => handleTranslate()}
                disabled={isTranslating || !inputText.trim()}
                style={[
                  s.translateBtn,
                  (!inputText.trim() || isTranslating) && s.translateBtnDisabled,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Translate to ASL"
              >
                {isTranslating ? (
                  <ActivityIndicator size="small" color="#07110F" />
                ) : (
                  <Ionicons name="sparkles" size={18} color="#07110F" />
                )}
              </Pressable>
            </View>

            {isListeningMic && (
              <View style={s.listeningBanner}>
                <View style={s.listeningDot} />
                <Text style={s.listeningText}>Listening for English speech...</Text>
              </View>
            )}

            {micError && <Text style={s.micErrorText}>{micError}</Text>}
          </View>

          {/* Quick Preset Chips */}
          <View style={s.presetsSection}>
            <Text style={s.presetsLabel}>CAMPUS & CONVERSATION PRESETS:</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={s.presetsRow}
            >
              {PRESET_PHRASES.map((phrase) => {
                const isSelected = selectedPreset === phrase;
                return (
                  <Pressable
                    key={phrase}
                    onPress={() => handleSelectPreset(phrase)}
                    style={[s.presetChip, isSelected && s.presetChipSelected]}
                    accessibilityRole="button"
                  >
                    <Text
                      style={[s.presetChipText, isSelected && s.presetChipTextSelected]}
                    >
                      {phrase}
                    </Text>
                  </Pressable>
                );
              })}
            </ScrollView>
          </View>

          {/* ASL Grammar Transformation Breakdown */}
          <View style={s.aslGlossCard}>
            <View style={s.aslGlossHeader}>
              <View style={s.glossTitleRow}>
                <Ionicons name="git-branch-outline" size={15} color="#C8FF72" />
                <Text style={s.aslGlossHeaderTitle}>ASL GRAMMAR & GLOSS SYNTAX</Text>
              </View>
              <View style={s.tokenCountBadge}>
                <Text style={s.tokenCountText}>{result.tokens.length} signs</Text>
              </View>
            </View>

            <View style={s.glossSentenceRow}>
              <Text style={s.glossSentence} accessibilityLiveRegion="polite">
                {result.aslSentence || 'NO SIGNS'}
              </Text>
            </View>

            <View style={s.grammarNotesRow}>
              <Ionicons name="information-circle-outline" size={13} color="#94A9A0" />
              <Text style={s.grammarNotesText}>
                Omitted copulas (is/are/a) • WH-words reordered to end • Fingerspelling for names
              </Text>
            </View>
          </View>

          {/* Interactive Skeletal Sign Player */}
          <View style={s.playerCard}>
            <SignPlayer tokens={result.tokens} autoPlay={true} height={340} />
          </View>

          {/* 3D WebGL PBR Hand & Linguistic Guide Info */}
          <View style={s.legendCard}>
            <View style={s.avatarBadgeRow}>
              <View style={s.avatarActivePill}>
                <Ionicons name="cube" size={13} color="#07110F" />
                <Text style={s.avatarActivePillText}>Three.js WebGL 3D Studio</Text>
              </View>
              <Text style={s.nmmTag}>PBR Lighting & 360° Drag Orbit</Text>
            </View>

            <Text style={s.avatarDesc}>
              Hardware-accelerated 3D WebGL hand rendering with real-time specular highlights,
              anatomical finger tapering, depth sorting, and complete linguistic ASL execution guides.
            </Text>

            <View style={s.legendItems}>
              <View style={s.legendItem}>
                <View style={[s.legendDot, { backgroundColor: '#B7FF56' }]} />
                <Text style={s.legendText}>360° Real-time Orbit Drag</Text>
              </View>
              <View style={s.legendItem}>
                <View style={[s.legendDot, { backgroundColor: '#00E5FF' }]} />
                <Text style={s.legendText}>Handshape & Palm Orientation</Text>
              </View>
              <View style={s.legendItem}>
                <View style={[s.legendDot, { backgroundColor: '#FFD043' }]} />
                <Text style={s.legendText}>Physical Trajectory Guide</Text>
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#07110F',
  },
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#162E25',
    backgroundColor: '#07110F',
  },
  backBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#0E211A',
    borderWidth: 1,
    borderColor: '#1C3B2E',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  headerTitleWrap: {
    flex: 1,
  },
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  headerTitle: {
    color: '#F1F5F3',
    fontSize: 17,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  headerSubtitle: {
    color: '#94A9A0',
    fontSize: 11,
    fontWeight: '500',
    marginTop: 1,
  },
  modeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0D2A1C',
    borderWidth: 1,
    borderColor: '#1F563B',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
    gap: 6,
  },
  modeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#B7FF56',
  },
  modeBadgeText: {
    color: '#B7FF56',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
    gap: 16,
  },
  directionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0A1713',
    borderRadius: 14,
    padding: 10,
    borderWidth: 1,
    borderColor: '#162E25',
    gap: 8,
  },
  directionPillActive: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#B7FF56',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 10,
    gap: 6,
  },
  directionPillActiveText: {
    color: '#07110F',
    fontSize: 12,
    fontWeight: '800',
  },
  directionArrow: {
    paddingHorizontal: 2,
  },
  directionPillTarget: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0E211A',
    borderWidth: 1,
    borderColor: '#1C3B2E',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 10,
    gap: 6,
  },
  directionPillTargetText: {
    color: '#C8FF72',
    fontSize: 12,
    fontWeight: '700',
  },
  switchModeBtn: {
    marginLeft: 'auto',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#12261E',
    borderWidth: 1,
    borderColor: '#244D3C',
    paddingHorizontal: 9,
    paddingVertical: 6,
    borderRadius: 10,
    gap: 5,
  },
  switchModeBtnText: {
    color: '#C8FF72',
    fontSize: 11,
    fontWeight: '700',
  },
  inputCard: {
    backgroundColor: '#0C1B16',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#1A382C',
    padding: 14,
    gap: 10,
  },
  inputHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  inputHeaderLabel: {
    color: '#8CA197',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  clearTextBtn: {
    color: '#FF7D7D',
    fontSize: 11,
    fontWeight: '700',
  },
  textInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  textInput: {
    flex: 1,
    backgroundColor: '#07110F',
    borderWidth: 1,
    borderColor: '#1C3B2E',
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: '#F1F5F3',
    fontSize: 15,
    fontWeight: '600',
  },
  micBtn: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#0E211A',
    borderWidth: 1,
    borderColor: '#1C3B2E',
    alignItems: 'center',
    justifyContent: 'center',
  },
  micBtnActive: {
    backgroundColor: '#FFD043',
    borderColor: '#FFD043',
  },
  translateBtn: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#B7FF56',
    alignItems: 'center',
    justifyContent: 'center',
  },
  translateBtnDisabled: {
    opacity: 0.45,
  },
  listeningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 208, 67, 0.12)',
    borderWidth: 1,
    borderColor: '#FFD043',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
  listeningDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FFD043',
  },
  listeningText: {
    color: '#FFD043',
    fontSize: 12,
    fontWeight: '700',
  },
  micErrorText: {
    color: '#FF8A8A',
    fontSize: 11,
    marginTop: 2,
  },
  presetsSection: {
    gap: 8,
  },
  presetsLabel: {
    color: '#8CA197',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  presetsRow: {
    gap: 8,
    paddingVertical: 2,
  },
  presetChip: {
    backgroundColor: '#0C1B16',
    borderWidth: 1,
    borderColor: '#1A382C',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  presetChipSelected: {
    backgroundColor: '#173628',
    borderColor: '#B7FF56',
  },
  presetChipText: {
    color: '#D8E5DF',
    fontSize: 12,
    fontWeight: '600',
  },
  presetChipTextSelected: {
    color: '#B7FF56',
    fontWeight: '800',
  },
  aslGlossCard: {
    backgroundColor: '#0C1B16',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#1C3B2E',
    padding: 14,
    gap: 10,
  },
  aslGlossHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  glossTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  aslGlossHeaderTitle: {
    color: '#C8FF72',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  tokenCountBadge: {
    backgroundColor: '#07110F',
    borderWidth: 1,
    borderColor: '#1C3B2E',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  tokenCountText: {
    color: '#94A9A0',
    fontSize: 10,
    fontWeight: '700',
  },
  glossSentenceRow: {
    backgroundColor: '#07110F',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: '#162E25',
  },
  glossSentence: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '900',
    letterSpacing: 1.2,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  grammarNotesRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  grammarNotesText: {
    color: '#8CA197',
    fontSize: 11,
    fontWeight: '500',
  },
  playerCard: {
    backgroundColor: '#0C1B16',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#1C3B2E',
    overflow: 'hidden',
  },
  legendCard: {
    backgroundColor: '#091511',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#162E25',
    padding: 12,
    gap: 8,
  },
  avatarBadgeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  avatarActivePill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#B7FF56',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    gap: 5,
  },
  avatarActivePillText: {
    color: '#07110F',
    fontSize: 10,
    fontWeight: '800',
  },
  nmmTag: {
    color: '#C8FF72',
    fontSize: 10,
    fontWeight: '700',
  },
  avatarDesc: {
    color: '#8EA49B',
    fontSize: 12,
    lineHeight: 16,
  },
  legendTitle: {
    color: '#667C73',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  legendItems: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    color: '#94A9A0',
    fontSize: 11,
    fontWeight: '600',
  },
});

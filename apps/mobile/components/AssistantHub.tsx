import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useState } from 'react';
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

type ActiveTab = 'hub' | 'settings' | 'telemetry';
type SelectedMode = 'scene' | 'sign';

const isElectron =
  typeof window !== 'undefined' &&
  (navigator.userAgent.toLowerCase().includes('electron') ||
    Boolean((window as any).process?.versions?.electron));

export default function AssistantHub() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('hub');
  const [selectedMode, setSelectedMode] = useState<SelectedMode>('scene');

  // Settings State
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [speechRate, setSpeechRate] = useState<'0.9' | '1.0' | '1.2'>('1.0');
  const [sensitivity, setSensitivity] = useState<'high' | 'balanced' | 'strict'>('balanced');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [pingStatus, setPingStatus] = useState<'idle' | 'testing' | 'online' | 'offline'>('idle');
  const [pingLatency, setPingLatency] = useState<number | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.localStorage) {
      const storedSpeech = window.localStorage.getItem('guidelens_speech');
      if (storedSpeech !== null) setSpeechEnabled(storedSpeech === 'true');

      const storedRate = window.localStorage.getItem('guidelens_speech_rate');
      if (storedRate === '0.9' || storedRate === '1.0' || storedRate === '1.2') {
        setSpeechRate(storedRate);
      }

      const storedSens = window.localStorage.getItem('guidelens_sensitivity');
      if (storedSens === 'high' || storedSens === 'balanced' || storedSens === 'strict') {
        setSensitivity(storedSens);
      }

      const storedUrl = window.localStorage.getItem('guidelens_api_url');
      if (storedUrl) setApiUrl(storedUrl);
    }
  }, []);

  const saveSetting = (key: string, value: string) => {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(`guidelens_${key}`, value);
    }
  };

  const toggleSpeech = () => {
    const next = !speechEnabled;
    setSpeechEnabled(next);
    saveSetting('speech', String(next));
    if (next) {
      Speech.speak('Voice narration enabled', { rate: Number(speechRate) });
    } else {
      Speech.stop();
    }
  };

  const changeRate = (rate: '0.9' | '1.0' | '1.2') => {
    setSpeechRate(rate);
    saveSetting('speech_rate', rate);
    if (speechEnabled) {
      Speech.speak(`Speech speed set to ${rate}x`, { rate: Number(rate) });
    }
  };

  const changeSensitivity = (val: 'high' | 'balanced' | 'strict') => {
    setSensitivity(val);
    saveSetting('sensitivity', val);
  };

  const testConnection = async () => {
    setPingStatus('testing');
    setPingLatency(null);
    const start = performance.now();
    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, '')}/healthz`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      const duration = Math.round(performance.now() - start);
      if (response.ok) {
        setPingStatus('online');
        setPingLatency(duration);
      } else {
        setPingStatus('offline');
      }
    } catch {
      setPingStatus('offline');
    }
  };

  const launchCamera = (modeToLaunch?: SelectedMode) => {
    const target = modeToLaunch || selectedMode;
    router.push({
      pathname: '/camera',
      params: { initialMode: target },
    });
  };

  return (
    <SafeAreaView style={s.safe}>
      {/* Top Header / Brand Bar */}
      <View style={s.header}>
        <View style={s.brandGroup}>
          <View style={s.logoBadge}>
            <Ionicons name="scan" size={20} color="#07110F" />
          </View>
          <View>
            <View style={s.brandTitleRow}>
              <Text style={s.brandTitle}>GuideLens</Text>
              <View style={s.versionChip}>
                <Text style={s.versionText}>v0.1.4</Text>
              </View>
            </View>
            <Text style={s.brandSubtitle}>Spatial Perception & Sign Intelligence</Text>
          </View>
        </View>

        <View style={s.headerRight}>
          {Platform.OS === 'web' && !isElectron && (
            <Pressable
              style={s.webLandingBtn}
              onPress={() => router.push('/')}
              accessibilityRole="button"
              accessibilityLabel="View landing page overview"
            >
              <Ionicons name="globe-outline" size={13} color="#B7FF56" />
              <Text style={s.webLandingBtnText}>Overview</Text>
            </Pressable>
          )}
          <View style={s.statusPill}>
            <View style={s.statusGlowDot} />
            <Text style={s.statusPillText}>ON-DEVICE AI READY</Text>
          </View>
        </View>
      </View>

      {/* Primary Navigation Tabs */}
      <View style={s.tabBar} accessibilityRole="tablist">
        <Pressable
          accessibilityRole="tab"
          accessibilityState={{ selected: activeTab === 'hub' }}
          onPress={() => setActiveTab('hub')}
          style={[s.tabButton, activeTab === 'hub' && s.tabButtonActive]}
        >
          <Ionicons
            name={activeTab === 'hub' ? 'apps' : 'apps-outline'}
            size={16}
            color={activeTab === 'hub' ? '#07110F' : '#9EB5AA'}
          />
          <Text style={[s.tabButtonText, activeTab === 'hub' && s.tabButtonTextActive]}>
            Assistant Hub
          </Text>
        </Pressable>

        <Pressable
          accessibilityRole="tab"
          accessibilityState={{ selected: activeTab === 'settings' }}
          onPress={() => setActiveTab('settings')}
          style={[s.tabButton, activeTab === 'settings' && s.tabButtonActive]}
        >
          <Ionicons
            name={activeTab === 'settings' ? 'settings' : 'settings-outline'}
            size={16}
            color={activeTab === 'settings' ? '#07110F' : '#9EB5AA'}
          />
          <Text style={[s.tabButtonText, activeTab === 'settings' && s.tabButtonTextActive]}>
            Settings
          </Text>
        </Pressable>

        <Pressable
          accessibilityRole="tab"
          accessibilityState={{ selected: activeTab === 'telemetry' }}
          onPress={() => setActiveTab('telemetry')}
          style={[s.tabButton, activeTab === 'telemetry' && s.tabButtonActive]}
        >
          <Ionicons
            name={activeTab === 'telemetry' ? 'hardware-chip' : 'hardware-chip-outline'}
            size={16}
            color={activeTab === 'telemetry' ? '#07110F' : '#9EB5AA'}
          />
          <Text style={[s.tabButtonText, activeTab === 'telemetry' && s.tabButtonTextActive]}>
            Telemetry
          </Text>
        </Pressable>
      </View>

      <ScrollView contentContainerStyle={s.contentScroll} showsVerticalScrollIndicator={false}>
        {/* TAB 1: ASSISTANT HUB */}
        {activeTab === 'hub' && (
          <View style={s.tabContent}>
            {/* Hero Launchpad Cockpit Card */}
            <View style={s.heroCard}>
              <View style={s.heroGlowCorner} />
              <View style={s.heroHeader}>
                <View style={s.heroBadge}>
                  <Ionicons name="sparkles" size={13} color="#B7FF56" />
                  <Text style={s.heroBadgeText}>ZERO LATENCY · ON-DEVICE</Text>
                </View>
                <Text style={s.heroTitle}>Visual & Sign Intelligence</Text>
                <Text style={s.heroDesc}>
                  Private, real-time spatial awareness and ASL translation powered by on-device neural
                  networks.
                </Text>
              </View>

              {/* Mode Select Segment within Hero */}
              <View style={s.modePickerContainer}>
                <Text style={s.modePickerLabel}>SELECT STARTING MODE</Text>
                <View style={s.modePicker}>
                  <Pressable
                    onPress={() => setSelectedMode('scene')}
                    style={[s.modeOption, selectedMode === 'scene' && s.modeOptionActive]}
                  >
                    <Ionicons
                      name="eye"
                      size={18}
                      color={selectedMode === 'scene' ? '#B7FF56' : '#88A397'}
                    />
                    <View>
                      <Text
                        style={[
                          s.modeOptionTitle,
                          selectedMode === 'scene' && s.modeOptionTitleActive,
                        ]}
                      >
                        Scene Assist
                      </Text>
                      <Text style={s.modeOptionSub}>Objects, obstacles & distance</Text>
                    </View>
                  </Pressable>

                  <Pressable
                    onPress={() => setSelectedMode('sign')}
                    style={[s.modeOption, selectedMode === 'sign' && s.modeOptionActive]}
                  >
                    <Ionicons
                      name="hand-left"
                      size={18}
                      color={selectedMode === 'sign' ? '#B7FF56' : '#88A397'}
                    />
                    <View>
                      <Text
                        style={[
                          s.modeOptionTitle,
                          selectedMode === 'sign' && s.modeOptionTitleActive,
                        ]}
                      >
                        Sign to Voice
                      </Text>
                      <Text style={s.modeOptionSub}>ASL gestures & letters</Text>
                    </View>
                  </Pressable>
                </View>
              </View>

              {/* Main Launch CTA Button */}
              <Pressable
                onPress={() => launchCamera()}
                style={s.launchButton}
                accessibilityRole="button"
                accessibilityLabel="Launch Vision Camera"
              >
                <View style={s.launchButtonIconWrapper}>
                  <Ionicons name="camera" size={22} color="#07110F" />
                </View>
                <View style={s.launchButtonTextCol}>
                  <Text style={s.launchButtonMainText}>
                    Launch {selectedMode === 'scene' ? 'Scene Assist' : 'Sign Interpreter'}
                  </Text>
                  <Text style={s.launchButtonSubText}>Opens camera with local neural engine</Text>
                </View>
                <Ionicons name="arrow-forward" size={20} color="#07110F" />
              </Pressable>
            </View>

            {/* Quick Action Mode Cards */}
            <View style={s.sectionHeader}>
              <Text style={s.sectionTitle}>Perception Capabilities</Text>
              <Text style={s.sectionDesc}>Direct access to specialized vision pipelines</Text>
            </View>

            <View style={s.cardsGrid}>
              {/* Card 1: Scene Assist */}
              <View style={s.featureCard}>
                <View style={s.cardTopRow}>
                  <View style={s.cardIconCircle}>
                    <Ionicons name="eye-outline" size={22} color="#B7FF56" />
                  </View>
                  <View style={s.cardTag}>
                    <Text style={s.cardTagText}>YOLOX-Nano · 3.7 MB</Text>
                  </View>
                </View>

                <Text style={s.cardTitle}>Spatial Scene Awareness</Text>
                <Text style={s.cardBody}>
                  Recognizes 80 categories of objects and persons, measures visual distance, and
                  announces nearby items via synthesized speech.
                </Text>

                <Pressable
                  onPress={() => launchCamera('scene')}
                  style={s.cardActionBtn}
                  accessibilityLabel="Open Scene Assist"
                >
                  <Text style={s.cardActionBtnText}>Open Scene Assist</Text>
                  <Ionicons name="chevron-forward" size={16} color="#B7FF56" />
                </Pressable>
              </View>

              {/* Card 2: Sign to Voice */}
              <View style={s.featureCard}>
                <View style={s.cardTopRow}>
                  <View style={[s.cardIconCircle, s.cardIconCircleSign]}>
                    <Ionicons name="hand-left-outline" size={22} color="#C8FF72" />
                  </View>
                  <View style={[s.cardTag, s.cardTagSign]}>
                    <Text style={[s.cardTagText, s.cardTagTextSign]}>Transformer + Kinematics</Text>
                  </View>
                </View>

                <Text style={s.cardTitle}>ASL Sign to Voice</Text>
                <Text style={s.cardBody}>
                  Instant kinematic gesture recognition for conversational phrases (Hello, Yes, No,
                  Please, Thank You, Water) plus 250-word isolated ASL Transformer.
                </Text>

                <Pressable
                  onPress={() => launchCamera('sign')}
                  style={[s.cardActionBtn, s.cardActionBtnSign]}
                  accessibilityLabel="Open Sign Interpreter"
                >
                  <Text style={[s.cardActionBtnText, s.cardActionBtnTextSign]}>
                    Open Sign Interpreter
                  </Text>
                  <Ionicons name="chevron-forward" size={16} color="#C8FF72" />
                </Pressable>
              </View>
            </View>

            {/* Quick Status Bar */}
            <View style={s.telemetryPillsRow}>
              <View style={s.telemetryPill}>
                <Ionicons name="shield-checkmark-outline" size={15} color="#94A9A0" />
                <Text style={s.telemetryPillLabel}>100% On-Device Privacy</Text>
              </View>
              <View style={s.telemetryPill}>
                <Ionicons name="flash-outline" size={15} color="#94A9A0" />
                <Text style={s.telemetryPillLabel}>WASM SIMD Acceleration</Text>
              </View>
              <View style={s.telemetryPill}>
                <Ionicons name="volume-high-outline" size={15} color="#94A9A0" />
                <Text style={s.telemetryPillLabel}>Native Speech Synthesizer</Text>
              </View>
            </View>
          </View>
        )}

        {/* TAB 2: SETTINGS */}
        {activeTab === 'settings' && (
          <View style={s.tabContent}>
            <View style={s.settingsContainer}>
              <View style={s.settingsHeader}>
                <Text style={s.settingsHeading}>App Preferences</Text>
                <Text style={s.settingsSub}>
                  Customize speech audio feedback, neural sensitivity, and optional DGX server.
                </Text>
              </View>

              {/* Group 1: Voice & Speech */}
              <View style={s.settingsCard}>
                <View style={s.settingsCardHeader}>
                  <Ionicons name="volume-high-outline" size={18} color="#B7FF56" />
                  <Text style={s.settingsCardTitle}>Voice & Audio Narration</Text>
                </View>

                {/* Spoken Voice Toggle */}
                <View style={s.settingRow}>
                  <View style={s.settingCol}>
                    <Text style={s.settingLabel}>Spoken Voice Feedback</Text>
                    <Text style={s.settingDesc}>
                      Announces detected objects and translated sign words aloud.
                    </Text>
                  </View>
                  <Pressable
                    onPress={toggleSpeech}
                    style={[s.toggleBtn, speechEnabled && s.toggleBtnActive]}
                  >
                    <View style={[s.toggleKnob, speechEnabled && s.toggleKnobActive]} />
                  </Pressable>
                </View>

                {/* Speech Speed */}
                <View style={[s.settingRow, s.settingRowBorder]}>
                  <View style={s.settingCol}>
                    <Text style={s.settingLabel}>Speech Rate</Text>
                    <Text style={s.settingDesc}>Pacing for spoken announcements.</Text>
                  </View>
                  <View style={s.segmentedRow}>
                    {(['0.9', '1.0', '1.2'] as const).map((rate) => (
                      <Pressable
                        key={rate}
                        onPress={() => changeRate(rate)}
                        style={[s.segmentedOption, speechRate === rate && s.segmentedOptionActive]}
                      >
                        <Text
                          style={[
                            s.segmentedOptionText,
                            speechRate === rate && s.segmentedOptionTextActive,
                          ]}
                        >
                          {rate}x
                        </Text>
                      </Pressable>
                    ))}
                  </View>
                </View>
              </View>

              {/* Group 2: Vision Sensitivity */}
              <View style={s.settingsCard}>
                <View style={s.settingsCardHeader}>
                  <Ionicons name="scan-outline" size={18} color="#B7FF56" />
                  <Text style={s.settingsCardTitle}>Vision & Neural Confidence</Text>
                </View>

                <View style={s.settingRow}>
                  <View style={s.settingCol}>
                    <Text style={s.settingLabel}>Object Detection Sensitivity</Text>
                    <Text style={s.settingDesc}>
                      Confidence threshold for local YOLOX bounding boxes.
                    </Text>
                  </View>
                  <View style={s.segmentedRow}>
                    {(
                      [
                        { key: 'high', label: 'High (0.30)' },
                        { key: 'balanced', label: 'Balanced (0.38)' },
                        { key: 'strict', label: 'Strict (0.50)' },
                      ] as const
                    ).map((item) => (
                      <Pressable
                        key={item.key}
                        onPress={() => changeSensitivity(item.key)}
                        style={[
                          s.segmentedOption,
                          sensitivity === item.key && s.segmentedOptionActive,
                        ]}
                      >
                        <Text
                          style={[
                            s.segmentedOptionText,
                            sensitivity === item.key && s.segmentedOptionTextActive,
                          ]}
                        >
                          {item.label}
                        </Text>
                      </Pressable>
                    ))}
                  </View>
                </View>
              </View>

              {/* Group 3: Optional DGX Spark Cloud Endpoint */}
              <View style={s.settingsCard}>
                <View style={s.settingsCardHeader}>
                  <Ionicons name="cloud-outline" size={18} color="#B7FF56" />
                  <Text style={s.settingsCardTitle}>DGX Spark Vision Server (Optional)</Text>
                </View>
                <Text style={s.settingsCardNote}>
                  GuideLens runs completely offline. You can optionally connect a DGX Spark server for
                  extended visual Q&A and continuous sign language narration.
                </Text>

                <View style={s.apiUrlRow}>
                  <TextInput
                    value={apiUrl}
                    onChangeText={(val) => {
                      setApiUrl(val);
                      saveSetting('api_url', val);
                      setPingStatus('idle');
                    }}
                    placeholder="http://localhost:8000"
                    placeholderTextColor="#5C7A6E"
                    style={s.apiInput}
                    autoCapitalize="none"
                    autoCorrect={false}
                  />
                  <Pressable
                    onPress={testConnection}
                    style={s.pingBtn}
                    disabled={pingStatus === 'testing'}
                  >
                    <Text style={s.pingBtnText}>
                      {pingStatus === 'testing' ? 'Testing…' : 'Ping Server'}
                    </Text>
                  </Pressable>
                </View>

                {pingStatus !== 'idle' && (
                  <View style={s.pingResultRow}>
                    <Ionicons
                      name={pingStatus === 'online' ? 'checkmark-circle' : 'close-circle'}
                      size={17}
                      color={pingStatus === 'online' ? '#B7FF56' : '#FF7865'}
                    />
                    <Text
                      style={[
                        s.pingResultText,
                        pingStatus === 'online' ? s.pingResultOnline : s.pingResultOffline,
                      ]}
                    >
                      {pingStatus === 'online'
                        ? `Connected (${pingLatency}ms latency)`
                        : 'Server unreachable (Offline mode active)'}
                    </Text>
                  </View>
                )}
              </View>
            </View>
          </View>
        )}

        {/* TAB 3: TELEMETRY & SPECS */}
        {activeTab === 'telemetry' && (
          <View style={s.tabContent}>
            <View style={s.settingsContainer}>
              <View style={s.settingsHeader}>
                <Text style={s.settingsHeading}>Neural Engine Telemetry</Text>
                <Text style={s.settingsSub}>
                  On-device models, runtime specifications, and pipeline topology.
                </Text>
              </View>

              <View style={s.specCard}>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Object Detector</Text>
                  <Text style={s.specVal}>YOLOX-Nano (ONNX WASM / WebGL)</Text>
                </View>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Object Model Size</Text>
                  <Text style={s.specVal}>3.7 MB (416×416 input, 80 COCO classes)</Text>
                </View>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Sign Language Model</Text>
                  <Text style={s.specVal}>Google ASL Transformer (64-frame context)</Text>
                </View>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Sign Vocabulary</Text>
                  <Text style={s.specVal}>250 isolated ASL signs + 6 instant gestures</Text>
                </View>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Kinematic Gestures</Text>
                  <Text style={s.specVal}>Hello, Yes, No, Please, Thank You, Water</Text>
                </View>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Landmark Engine</Text>
                  <Text style={s.specVal}>MediaPipe Hands (21 3D pts) + Face + Pose</Text>
                </View>
                <View style={s.specRow}>
                  <Text style={s.specKey}>Privacy Level</Text>
                  <Text style={s.specValHighlight}>100% Zero-Telemetry · Air-Gapped</Text>
                </View>
              </View>

              <View style={s.versionFooter}>
                <Text style={s.versionFooterText}>
                  GuideLens Native Edition · Built for SFBU × NVIDIA DGX Hackathon 2026
                </Text>
              </View>
            </View>
          </View>
        )}
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
    paddingHorizontal: 24,
    paddingTop: 16,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.07)',
  },
  brandGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  logoBadge: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: '#B7FF56',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#B7FF56',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.35,
    shadowRadius: 8,
  },
  brandTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  brandTitle: {
    color: '#F4FAF6',
    fontSize: 19,
    fontWeight: '800',
    letterSpacing: -0.4,
  },
  versionChip: {
    backgroundColor: 'rgba(183, 255, 86, 0.12)',
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.25)',
  },
  versionText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '700',
  },
  brandSubtitle: {
    color: '#82998F',
    fontSize: 11,
    fontWeight: '500',
    marginTop: 1,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  webLandingBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.25)',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 999,
  },
  webLandingBtnText: {
    color: '#DDE9E2',
    fontSize: 12,
    fontWeight: '600',
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    backgroundColor: 'rgba(183, 255, 86, 0.09)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.28)',
    borderRadius: 99,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  statusGlowDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: '#B7FF56',
    shadowColor: '#B7FF56',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 5,
  },
  statusPillText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },

  // Tab Bar
  tabBar: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    paddingVertical: 10,
    gap: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.05)',
    backgroundColor: '#07110F',
  },
  tabButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
  },
  tabButtonActive: {
    backgroundColor: '#B7FF56',
  },
  tabButtonText: {
    color: '#8CA499',
    fontSize: 12,
    fontWeight: '700',
  },
  tabButtonTextActive: {
    color: '#07110F',
    fontWeight: '800',
  },

  contentScroll: {
    padding: 24,
    paddingBottom: 48,
    maxWidth: 1040,
    width: '100%',
    alignSelf: 'center',
  },
  tabContent: {
    gap: 24,
  },

  // Hero Card
  heroCard: {
    backgroundColor: 'rgba(10, 28, 21, 0.95)',
    borderRadius: 22,
    borderWidth: 1.5,
    borderColor: 'rgba(183, 255, 86, 0.35)',
    padding: 24,
    position: 'relative',
    overflow: 'hidden',
    shadowColor: '#B7FF56',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 16,
  },
  heroGlowCorner: {
    position: 'absolute',
    top: -50,
    right: -50,
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: 'rgba(183, 255, 86, 0.08)',
  },
  heroHeader: {
    gap: 8,
  },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(183, 255, 86, 0.12)',
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderRadius: 8,
  },
  heroBadgeText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.9,
  },
  heroTitle: {
    color: '#F4FAF6',
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: -0.6,
  },
  heroDesc: {
    color: '#95ADA1',
    fontSize: 14,
    lineHeight: 21,
    maxWidth: 620,
  },

  // Mode Picker inside Hero
  modePickerContainer: {
    marginTop: 20,
    gap: 8,
  },
  modePickerLabel: {
    color: '#6F8C7F',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  modePicker: {
    flexDirection: Platform.OS === 'web' ? 'row' : 'column',
    gap: 10,
  },
  modeOption: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: 'rgba(4, 15, 11, 0.7)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 14,
    padding: 14,
  },
  modeOptionActive: {
    borderColor: '#B7FF56',
    backgroundColor: 'rgba(183, 255, 86, 0.08)',
  },
  modeOptionTitle: {
    color: '#D4E4DC',
    fontSize: 14,
    fontWeight: '700',
  },
  modeOptionTitleActive: {
    color: '#F4FAF6',
    fontWeight: '800',
  },
  modeOptionSub: {
    color: '#7C988B',
    fontSize: 11,
    marginTop: 2,
  },

  // Main Launch Button
  launchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: '#B7FF56',
    borderRadius: 16,
    paddingHorizontal: 20,
    paddingVertical: 16,
    marginTop: 22,
    shadowColor: '#B7FF56',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
  },
  launchButtonIconWrapper: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: 'rgba(7, 17, 15, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  launchButtonTextCol: {
    flex: 1,
  },
  launchButtonMainText: {
    color: '#07110F',
    fontSize: 16,
    fontWeight: '800',
  },
  launchButtonSubText: {
    color: '#1A3328',
    fontSize: 11,
    fontWeight: '600',
    marginTop: 1,
  },

  // Section Headers
  sectionHeader: {
    marginTop: 6,
    gap: 4,
  },
  sectionTitle: {
    color: '#F2FAF6',
    fontSize: 18,
    fontWeight: '800',
  },
  sectionDesc: {
    color: '#7F9A8E',
    fontSize: 13,
  },

  // Capability Cards Grid
  cardsGrid: {
    flexDirection: Platform.OS === 'web' ? 'row' : 'column',
    gap: 16,
  },
  featureCard: {
    flex: 1,
    backgroundColor: 'rgba(9, 23, 17, 0.85)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 18,
    padding: 20,
    gap: 12,
  },
  cardTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  cardIconCircle: {
    width: 42,
    height: 42,
    borderRadius: 12,
    backgroundColor: 'rgba(183, 255, 86, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardIconCircleSign: {
    backgroundColor: 'rgba(200, 255, 114, 0.12)',
  },
  cardTag: {
    backgroundColor: 'rgba(183, 255, 86, 0.08)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  cardTagSign: {
    backgroundColor: 'rgba(200, 255, 114, 0.08)',
  },
  cardTagText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '700',
  },
  cardTagTextSign: {
    color: '#C8FF72',
  },
  cardTitle: {
    color: '#F4FAF6',
    fontSize: 17,
    fontWeight: '800',
  },
  cardBody: {
    color: '#8FAAA0',
    fontSize: 13,
    lineHeight: 19,
    flex: 1,
  },
  cardActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(183, 255, 86, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.25)',
    borderRadius: 11,
    paddingHorizontal: 14,
    paddingVertical: 11,
    marginTop: 4,
  },
  cardActionBtnSign: {
    backgroundColor: 'rgba(200, 255, 114, 0.08)',
    borderColor: 'rgba(200, 255, 114, 0.25)',
  },
  cardActionBtnText: {
    color: '#B7FF56',
    fontSize: 12,
    fontWeight: '800',
  },
  cardActionBtnTextSign: {
    color: '#C8FF72',
  },

  // Telemetry row
  telemetryPillsRow: {
    flexDirection: Platform.OS === 'web' ? 'row' : 'column',
    gap: 10,
    marginTop: 4,
  },
  telemetryPill: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.06)',
    borderRadius: 12,
    paddingVertical: 11,
    paddingHorizontal: 14,
  },
  telemetryPillLabel: {
    color: '#8FA99E',
    fontSize: 11,
    fontWeight: '700',
  },

  // Settings Screen
  settingsContainer: {
    gap: 20,
  },
  settingsHeader: {
    gap: 4,
  },
  settingsHeading: {
    color: '#F4FAF6',
    fontSize: 22,
    fontWeight: '800',
  },
  settingsSub: {
    color: '#849F93',
    fontSize: 13,
  },
  settingsCard: {
    backgroundColor: 'rgba(8, 22, 16, 0.85)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 18,
    padding: 20,
    gap: 16,
  },
  settingsCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
    paddingBottom: 12,
  },
  settingsCardTitle: {
    color: '#F4FAF6',
    fontSize: 15,
    fontWeight: '800',
  },
  settingsCardNote: {
    color: '#829C90',
    fontSize: 12,
    lineHeight: 18,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
  },
  settingRowBorder: {
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.05)',
    paddingTop: 16,
  },
  settingCol: {
    flex: 1,
  },
  settingLabel: {
    color: '#E8F3EE',
    fontSize: 14,
    fontWeight: '700',
  },
  settingDesc: {
    color: '#769286',
    fontSize: 11,
    marginTop: 2,
  },
  toggleBtn: {
    width: 48,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(255, 255, 255, 0.12)',
    padding: 3,
    justifyContent: 'center',
  },
  toggleBtnActive: {
    backgroundColor: '#B7FF56',
  },
  toggleKnob: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#7A9188',
  },
  toggleKnobActive: {
    backgroundColor: '#07110F',
    alignSelf: 'flex-end',
  },
  segmentedRow: {
    flexDirection: 'row',
    backgroundColor: 'rgba(4, 15, 11, 0.7)',
    borderRadius: 10,
    padding: 3,
    gap: 3,
  },
  segmentedOption: {
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 8,
  },
  segmentedOptionActive: {
    backgroundColor: '#B7FF56',
  },
  segmentedOptionText: {
    color: '#8EA79B',
    fontSize: 11,
    fontWeight: '700',
  },
  segmentedOptionTextActive: {
    color: '#07110F',
    fontWeight: '800',
  },

  // API Url row
  apiUrlRow: {
    flexDirection: Platform.OS === 'web' ? 'row' : 'column',
    gap: 8,
  },
  apiInput: {
    flex: 1,
    backgroundColor: 'rgba(4, 15, 11, 0.8)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    color: '#EAF5EF',
    fontSize: 13,
  },
  pingBtn: {
    backgroundColor: 'rgba(183, 255, 86, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.35)',
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pingBtnText: {
    color: '#B7FF56',
    fontSize: 12,
    fontWeight: '800',
  },
  pingResultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
  },
  pingResultText: {
    fontSize: 12,
    fontWeight: '700',
  },
  pingResultOnline: { color: '#B7FF56' },
  pingResultOffline: { color: '#FF7865' },

  // Specs Card
  specCard: {
    backgroundColor: 'rgba(8, 22, 16, 0.85)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 18,
    padding: 20,
    gap: 12,
  },
  specRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.04)',
  },
  specKey: {
    color: '#849E93',
    fontSize: 12,
    fontWeight: '600',
  },
  specVal: {
    color: '#E2EFE9',
    fontSize: 12,
    fontWeight: '700',
  },
  specValHighlight: {
    color: '#B7FF56',
    fontSize: 12,
    fontWeight: '800',
  },
  versionFooter: {
    alignItems: 'center',
    marginTop: 10,
  },
  versionFooterText: {
    color: '#5C746B',
    fontSize: 11,
    fontWeight: '600',
  },
});

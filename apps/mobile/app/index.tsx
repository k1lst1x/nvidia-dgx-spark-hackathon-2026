import AssistantHub from '../components/AssistantHub';
import RoomSketchMap from '../components/RoomSketchMap';
import type { SightObservation } from '../src/campusMemory';
import { Ionicons } from '@expo/vector-icons';
import { Link } from 'expo-router';
import { ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  AccessibilityInfo,
  Animated,
  LayoutChangeEvent,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleProp,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const benefits = [
  {
    icon: 'git-network-outline' as const,
    eyebrow: 'SHARED SIGHT · CAMPUS MEMORY',
    title: 'A campus that remembers for blind students',
    text: 'Many phones feed one private DGX Spark. When any student spots a hazard or closed door, all blind students receive proactive spoken warnings before reaching it.',
  },
  {
    icon: 'git-compare-outline' as const,
    eyebrow: 'WHAT CHANGED? · DELTA SCAN',
    title: 'Hear only what is new, not the whole room',
    text: 'Blind users already know where walls and doors are. GuideLens compares live vision against room baselines to announce only moved obstacles and new items.',
  },
  {
    icon: 'people-outline' as const,
    eyebrow: 'DEAF–BLIND BRIDGE',
    title: 'Direct two-way conversation aid',
    text: 'Connects two people who could not talk directly: ASL gestures speak aloud to the blind partner, while spoken voice replies appear in high-contrast giant text with animated signs.',
  },
  {
    icon: 'swap-horizontal-outline' as const,
    eyebrow: 'ENGLISH ⇄ ASL SIGN STUDIO',
    title: 'Bidirectional sign language translation',
    text: 'Translates spoken or typed English into ASL grammar with animated 21-point skeletal hand gestures and instant fingerspelling fallback, offline-first.',
  },
  {
    icon: 'shield-checkmark-outline' as const,
    eyebrow: '100% ON-DEVICE PRIVACY',
    title: 'Runs with or without server',
    text: 'Zero-latency YOLOX object detection and MediaPipe ASL run privately on your processor with WASM SIMD acceleration and zero network dependency.',
  },
];

const pipeline = [
  {
    step: '01',
    icon: 'camera-outline' as const,
    label: 'Capture',
    title: 'One purposeful frame',
    text: 'The user points the phone at a scene and chooses what they need help with.',
    meta: 'Camera + task',
  },
  {
    step: '02',
    icon: 'resize-outline' as const,
    label: 'Prepare',
    title: 'A compact request',
    text: 'The app packages one JPEG with a concise prompt instead of streaming video.',
    meta: 'JPEG + context',
  },
  {
    step: '03',
    icon: 'hardware-chip-outline' as const,
    label: 'Reason',
    title: 'Vision on DGX Spark',
    text: 'A vision-language model interprets the image and returns only the requested detail.',
    meta: 'Qwen3-VL endpoint',
  },
  {
    step: '04',
    icon: 'chatbubble-ellipses-outline' as const,
    label: 'Respond',
    title: 'A calm spoken brief',
    text: 'The result is available as readable text and a concise voice response in the mobile app.',
    meta: 'Text + speech',
  },
];

// Sample pins for the landing-page map. They are labelled EXAMPLE on the page;
// in the app every pin is a report a person confirmed (see observation_store).
const EXAMPLE_ZONE = 'Library Corridor';
const EXAMPLE_PINS: Array<{ label: string; category: string; x: number; y: number; minutesAgo: number; confidence: number }> = [
  { label: 'wet floor sign', category: 'hazard', x: 0.3, y: 0.66, minutesAgo: 42, confidence: 0.91 },
  { label: 'backpack', category: 'obstacle', x: 0.52, y: 0.46, minutesAgo: 25, confidence: 0.86 },
  { label: 'chair', category: 'obstacle', x: 0.7, y: 0.3, minutesAgo: 12, confidence: 0.93 },
  { label: 'closed door', category: 'room_status', x: 0.84, y: 0.16, minutesAgo: 4, confidence: 0.88 },
];

function exampleObservations(now: number): SightObservation[] {
  return EXAMPLE_PINS.map((pin, index) => ({
    id: -(index + 1),
    category: pin.category,
    label: pin.label,
    description: `Example report: ${pin.label}`,
    zone_id: EXAMPLE_ZONE,
    timestamp: new Date(now - pin.minutesAgo * 60_000).toISOString(),
    source_device: 'example',
    source_session: 'example',
    confidence: pin.confidence,
    verification_status: 'verified',
    privacy_scope: 'shared',
    room_x: pin.x,
    room_y: pin.y,
    expires_at: new Date(now + 6 * 3_600_000).toISOString(),
    origin: 'manual',
  }));
}

const mapPoints = [
  { icon: 'checkmark-done-outline' as const, title: 'Confirmed before it is shared', text: 'The camera suggests; a person confirms. Unconfirmed guesses never reach other phones.' },
  { icon: 'time-outline' as const, title: 'Recent, then gone', text: 'Pins expire on their own, so the map shows what is true now, not last week.' },
  { icon: 'lock-closed-outline' as const, title: 'Personal stays personal', text: 'Private reminders, like where you left your keys, are never placed on the shared map.' },
];

const releaseDownloads = [
  {
    label: 'Android APK',
    detail: 'Universal ARM64 & ARMv7',
    icon: 'logo-android' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-Android.apk',
  },
  {
    label: 'Windows Installer',
    detail: '64-bit EXE (Setup)',
    icon: 'logo-windows' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-win-x64.exe',
  },
  {
    label: 'Windows Portable ZIP',
    detail: 'Zero-install standalone archive',
    icon: 'archive-outline' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-win-x64.zip',
  },
  {
    label: 'macOS Universal',
    detail: 'Apple Silicon & Intel DMG',
    icon: 'logo-apple' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-mac-universal.dmg',
  },
  {
    label: 'macOS Intel',
    detail: 'Dedicated x64 Intel Mac DMG',
    icon: 'hardware-chip-outline' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-mac-x64.dmg',
  },
  {
    label: 'Linux AppImage',
    detail: 'All Linux distributions',
    icon: 'logo-tux' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-linux-x86_64.AppImage',
  },
  {
    label: 'Linux DEB',
    detail: 'Ubuntu & Debian package',
    icon: 'cube-outline' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-linux-amd64.deb',
  },
  {
    label: 'Linux TAR.GZ',
    detail: 'Portable standalone archive',
    icon: 'archive-outline' as const,
    url: 'https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest/download/GuideLens-linux-x64.tar.gz',
  },
];

function useReducedMotion() {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled().then(setReducedMotion);
    const subscription = AccessibilityInfo.addEventListener('reduceMotionChanged', setReducedMotion);
    return () => subscription.remove();
  }, []);

  return reducedMotion;
}

function Reveal({
  children,
  scrollY,
  reducedMotion,
  style,
  delay = 0,
}: {
  children: ReactNode;
  scrollY: Animated.Value;
  reducedMotion: boolean;
  style?: StyleProp<ViewStyle>;
  delay?: number;
}) {
  const [top, setTop] = useState(1000);
  const onLayout = (event: LayoutChangeEvent) => setTop(event.nativeEvent.layout.y);

  if (reducedMotion) {
    return <View onLayout={onLayout} style={style}>{children}</View>;
  }

  const start = Math.max(0, top - 640 + delay);
  const end = start + 230;
  const opacity = scrollY.interpolate({
    inputRange: [start, end],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });
  const translateY = scrollY.interpolate({
    inputRange: [start, end],
    outputRange: [30, 0],
    extrapolate: 'clamp',
  });

  return (
    <Animated.View onLayout={onLayout} style={[style, { opacity, transform: [{ translateY }] }]}>
      {children}
    </Animated.View>
  );
}

function LandingPage() {
  const scrollY = useRef(new Animated.Value(0)).current;
  const reducedMotion = useReducedMotion();
  const examplePins = useMemo(() => exampleObservations(Date.now()), []);

  return (
    <SafeAreaView style={s.safe}>
      <Animated.ScrollView
        contentContainerStyle={s.page}
        onScroll={Animated.event([{ nativeEvent: { contentOffset: { y: scrollY } } }], { useNativeDriver: false })}
        scrollEventThrottle={16}
        showsVerticalScrollIndicator={false}
      >
        <View style={s.nav}>
          <View style={s.brandLockup}>
            <View style={s.brandMark}>
              <View style={s.brandRing} />
              <Ionicons name="eye-outline" size={19} color="#D9FF9B" />
            </View>
            <View>
              <Text style={s.brand}>GuideLens</Text>
              <Text style={s.brandSub}>VISUAL ASSISTANCE</Text>
            </View>
          </View>
          <View style={s.navRight}>
            <View style={s.statusPill} accessibilityLabel="1st Place Winner - SFBU NVIDIA DGX Spark Hackathon 2026">
              <View style={s.statusDot} />
              <Text style={s.statusText}>🏆 1ST PLACE WINNER · v1.0.0</Text>
            </View>
            <Link href="/hub" asChild>
              <Pressable accessibilityRole="button" accessibilityLabel="Launch Assistant Hub" style={s.navLaunchBtn}>
                <Ionicons name="sparkles" size={13} color="#07110F" />
                <Text style={s.navLaunchText}>Launch App</Text>
              </Pressable>
            </Link>
          </View>
        </View>

        <View style={s.hero}>
          <View style={s.eyebrow}>
            <View style={s.eyebrowLine} />
            <Text style={s.eyebrowText}>🏆 1ST PLACE WINNER · SFBU × NVIDIA DGX SPARK 2026</Text>
          </View>
          <Text style={s.title}>Visual help, delivered with clarity.</Text>
          <Text style={s.subtitle}>
            A voice-first companion for everyday visual questions: understand a scene, read a label, or ask about what is in front of you.
          </Text>
          <View style={s.actions}>
            <Link href="/hub" asChild>
              <Pressable accessibilityRole="button" accessibilityLabel="Launch the Web App Dashboard" style={s.primary}>
                <Ionicons name="sparkles" size={19} color="#0A120D" />
                <Text style={s.primaryText}>Launch Web App</Text>
                <Ionicons name="arrow-forward" size={17} color="#0A120D" />
              </Pressable>
            </Link>
            <Link href="/camera" asChild>
              <Pressable accessibilityRole="button" accessibilityLabel="Open live camera assistant" style={s.secondary}>
                <Ionicons name="camera-outline" size={19} color="#DDE9E2" />
                <Text style={s.secondaryText}>Camera Demo</Text>
              </Pressable>
            </Link>
            <Link href={"/translate" as any} asChild>
              <Pressable accessibilityRole="button" accessibilityLabel="Open English to ASL Sign Translator" style={s.secondary}>
                <Ionicons name="swap-horizontal" size={19} color="#C8FF72" />
                <Text style={[s.secondaryText, { color: '#C8FF72' }]}>ASL Studio</Text>
              </Pressable>
            </Link>
            <Pressable
              accessibilityRole="link"
              accessibilityLabel="Open the project on GitHub"
              onPress={() => Platform.OS === 'web' && window.open('https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026', '_blank')}
              style={s.secondary}
            >
              <Ionicons name="logo-github" size={19} color="#DDE9E2" />
              <Text style={s.secondaryText}>GitHub</Text>
            </Pressable>
            <Pressable
              accessibilityRole="link"
              accessibilityLabel="Download GuideLens applications"
              onPress={() => Platform.OS === 'web' && window.open('https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest', '_blank')}
              style={s.secondary}
            >
              <Ionicons name="download-outline" size={19} color="#DDE9E2" />
              <Text style={s.secondaryText}>Downloads</Text>
            </Pressable>
          </View>
        </View>

        <View style={s.downloadSection}>
          <View style={s.downloadHeading}>
            <View>
              <Text style={s.sectionOverline}>DOWNLOAD THE NATIVE APP</Text>
              <Text style={s.downloadTitle}>Run GuideLens on your device.</Text>
            </View>
            <Text style={s.downloadLead}>Local YOLOX-Nano object detection, ByteTrack tracker & MediaPipe ASL gestures are bundled in every native build. Choose your platform.</Text>
          </View>
          <View style={s.downloadGrid}>
            {releaseDownloads.map((release) => (
              <Pressable
                key={release.label}
                accessibilityRole="link"
                accessibilityLabel={`Download GuideLens for ${release.label}`}
                onPress={() => void Linking.openURL(release.url)}
                style={s.downloadCard}
              >
                <View style={s.downloadIcon}><Ionicons name={release.icon} size={22} color="#C8FF72" /></View>
                <View style={s.downloadCopy}>
                  <Text style={s.downloadLabel}>{release.label}</Text>
                  <Text style={s.downloadDetail}>{release.detail}</Text>
                </View>
                <Ionicons name="download-outline" size={20} color="#DDE9E2" />
              </Pressable>
            ))}
          </View>
          <Text style={s.downloadNote}>Production release builds (v1.0.0) may trigger an unsigned-app warning on desktop. Android users must allow installation from their browser.</Text>
        </View>

        <View style={s.briefing}>
          <View style={s.briefingHeader}>
            <View style={s.briefingLabel}>
              <View style={s.liveDot} />
              <Text style={s.briefingLabelText}>INTERFACE PREVIEW</Text>
            </View>
            <Text style={s.briefingTime}>ILLUSTRATIVE</Text>
          </View>
          <View style={s.briefingBody}>
            <View style={s.scenePane}>
              <View style={s.sceneGrid} />
              <View style={s.scanLine} />
              <View style={s.sceneTopline}>
                <Text style={s.frameLabel}>EXAMPLE CAMERA INPUT</Text>
                <Text style={s.frameLabel}>1 STILL FRAME</Text>
              </View>
              <View style={s.objectDoor}>
                <View style={s.objectTag}><Text style={s.objectTagText}>DOORWAY · 97%</Text></View>
              </View>
              <View style={s.objectChair}>
                <View style={s.objectTag}><Text style={s.objectTagText}>CHAIR · 93%</Text></View>
              </View>
              <View style={s.sceneFooter}>
                <Ionicons name="locate-outline" size={15} color="#C8FF72" />
                <Text style={s.sceneFooterText}>EXAMPLE OUTPUT</Text>
              </View>
            </View>
            <View style={s.responsePane}>
              <Text style={s.responseEyebrow}>EXAMPLE SPOKEN RESPONSE</Text>
              <Text style={s.responseTitle}>A clear route through the room.</Text>
              <Text style={s.responseText}>A doorway is ahead. A chair is on your right, about two steps away.</Text>
              <View style={s.waveRow} accessibilityLabel="Audio response waveform">
                {[14, 24, 35, 48, 31, 18, 40, 54, 36, 22, 31, 15].map((height, index) => (
                  <View key={index} style={[s.waveBar, { height }]} />
                ))}
              </View>
              <View style={s.responseMeta}>
                <View style={s.metaChip}><Ionicons name="volume-high-outline" size={14} color="#C8FF72" /><Text style={s.metaText}>READY TO SPEAK</Text></View>
                <Text style={s.metaTime}>00:02.4</Text>
              </View>
            </View>
          </View>
        </View>

        <Reveal scrollY={scrollY} reducedMotion={reducedMotion} style={s.introSection}>
          <Text style={s.sectionOverline}>BUILT FOR A FOCUSED MOMENT</Text>
          <Text style={s.sectionTitle}>A useful answer starts with a better question.</Text>
          <Text style={s.sectionLead}>GuideLens makes its response specific to the task instead of trying to narrate everything at once.</Text>
        </Reveal>

        <View style={s.cards}>
          {benefits.map((benefit, index) => (
            <Reveal key={benefit.title} scrollY={scrollY} reducedMotion={reducedMotion} delay={index * 55} style={s.cardWrap}>
              <View style={s.card}>
                <View style={s.cardTopline}>
                  <View style={s.cardIcon}><Ionicons name={benefit.icon} size={22} color="#C8FF72" /></View>
                  <Text style={s.cardEyebrow}>{benefit.eyebrow}</Text>
                </View>
                <Text style={s.cardTitle}>{benefit.title}</Text>
                <Text style={s.cardText}>{benefit.text}</Text>
              </View>
            </Reveal>
          ))}
        </View>

        <Reveal scrollY={scrollY} reducedMotion={reducedMotion} style={s.mapSection}>
          <View style={s.mapCopy}>
            <Text style={s.sectionOverline}>SHARED SIGHT · 3D SPATIAL MAP</Text>
            <Text style={s.mapTitle}>Reports stay where they were found.</Text>
            <Text style={s.mapLead}>
              When a student confirms something the camera saw, such as a wet floor, a closed door, or a moved chair, it is pinned to a shared sketch of that room for the next person who walks in.
            </Text>
            <View style={s.mapPoints}>
              {mapPoints.map((point) => (
                <View key={point.title} style={s.mapPoint}>
                  <View style={s.mapPointIcon}><Ionicons name={point.icon} size={17} color="#7EC8FF" /></View>
                  <View style={s.mapPointCopy}>
                    <Text style={s.mapPointTitle}>{point.title}</Text>
                    <Text style={s.mapPointText}>{point.text}</Text>
                  </View>
                </View>
              ))}
            </View>
            <Link href="/hub" asChild>
              <Pressable accessibilityRole="button" accessibilityLabel="Open Shared Sight in the app" style={s.mapCta}>
                <Ionicons name="git-network-outline" size={17} color="#07110F" />
                <Text style={s.mapCtaText}>Open Shared Sight</Text>
              </Pressable>
            </Link>
          </View>
          <View
            style={s.mapFrame}
            accessibilityLabel={`Example 3D map of the ${EXAMPLE_ZONE} with ${EXAMPLE_PINS.length} pinned reports: ${EXAMPLE_PINS.map((pin) => pin.label).join(', ')}.`}
          >
            <RoomSketchMap
              zoneId={EXAMPLE_ZONE}
              observations={examplePins}
              selectedPoint={null}
              onSelectPoint={() => {}}
              interactive={false}
              badge="EXAMPLE"
              autoRotate={!reducedMotion}
              description="Example pins in a sample room. In the app, each pin is a report a student confirmed, joined by the path they walked."
            />
          </View>
        </Reveal>

        <Reveal scrollY={scrollY} reducedMotion={reducedMotion} style={s.architecture}>
          <View style={s.architectureHeading}>
            <View>
              <Text style={s.sectionOverline}>THE INFERENCE PATH</Text>
              <Text style={s.architectureTitle}>From a single frame to a spoken answer.</Text>
            </View>
            <Text style={s.architectureSummary}>Designed for deliberate visual tasks, with a small request and an explicit response path.</Text>
          </View>

          <View style={s.pipeline}>
            {pipeline.map((item, index) => (
              <View key={item.step} style={s.pipelineItemWrap}>
                <View style={s.pipelineItem}>
                  <View style={s.pipelineTopline}>
                    <Text style={s.pipelineStep}>{item.step}</Text>
                    <Ionicons name={item.icon} size={21} color="#C8FF72" />
                  </View>
                  <Text style={s.pipelineLabel}>{item.label}</Text>
                  <Text style={s.pipelineTitle}>{item.title}</Text>
                  <Text style={s.pipelineText}>{item.text}</Text>
                  <View style={s.pipelineMeta}><Text style={s.pipelineMetaText}>{item.meta}</Text></View>
                </View>
                {index < pipeline.length - 1 && (
                  <View style={s.connector} accessibilityElementsHidden>
                    <View style={s.connectorLine} />
                    <Ionicons name={Platform.OS === 'web' ? 'arrow-forward' : 'arrow-down'} size={18} color="#779186" />
                  </View>
                )}
              </View>
            ))}
          </View>

          <View style={s.architectureFooter}>
            <Ionicons name="information-circle-outline" size={19} color="#A7BDB1" />
            <Text style={s.architectureFooterText}>YOLOX Nano performs real local object detection. A configured private backend adds concise vision-language descriptions without exposing its API key to the app.</Text>
          </View>
        </Reveal>

        <Reveal scrollY={scrollY} reducedMotion={reducedMotion} style={s.safety}>
          <View style={s.safetyIcon}><Ionicons name="shield-checkmark-outline" size={20} color="#DDE9E2" /></View>
          <View style={s.safetyCopy}>
            <Text style={s.safetyTitle}>Assistive, never authoritative.</Text>
            <Text style={s.safetyText}>GuideLens can be wrong and must not be used as the sole source for traffic, medication, emergencies, or mobility decisions.</Text>
          </View>
        </Reveal>

        <Text style={s.footer}>🏆 1ST PLACE WINNER · SFBU NVIDIA DGX SPARK HACKATHON 2026</Text>
      </Animated.ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#08110E' },
  page: { width: '100%', maxWidth: 1180, alignSelf: 'center', paddingHorizontal: 22, paddingBottom: 48 },
  nav: { minHeight: 82, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  brandLockup: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  brandMark: { width: 40, height: 40, borderRadius: 14, backgroundColor: '#14291F', borderWidth: 1, borderColor: '#365343', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
  brandRing: { position: 'absolute', width: 52, height: 52, borderRadius: 26, borderWidth: 1, borderColor: 'rgba(200,255,114,.20)' },
  brand: { color: '#F2F7F2', fontSize: 19, fontWeight: '800', letterSpacing: -0.4 },
  brandSub: { color: '#769086', fontSize: 9, fontWeight: '800', letterSpacing: 1.35, marginTop: 2 },
  navRight: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  navLaunchBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 7,
    backgroundColor: '#C8FF72',
    borderRadius: 99,
  },
  navLaunchText: {
    color: '#08110E',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  statusPill: { flexDirection: 'row', alignItems: 'center', gap: 7, paddingHorizontal: 11, paddingVertical: 8, borderWidth: 1, borderColor: '#2F4A3E', borderRadius: 99, backgroundColor: 'rgba(15,33,25,.72)' },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#C8FF72' },
  statusText: { color: '#BAD0C5', fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  hero: { paddingTop: 72, paddingBottom: 54, maxWidth: 830 },
  eyebrow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  eyebrowLine: { width: 30, height: 1, backgroundColor: '#C8FF72' },
  eyebrowText: { color: '#C8FF72', fontSize: 11, fontWeight: '800', letterSpacing: 1.65 },
  title: { color: '#F3F7F3', fontSize: Platform.OS === 'web' ? 72 : 51, lineHeight: Platform.OS === 'web' ? 76 : 56, fontWeight: '800', letterSpacing: -3.5, marginTop: 21, maxWidth: 760 },
  subtitle: { color: '#A8BCB2', fontSize: 19, lineHeight: 29, maxWidth: 660, marginTop: 22 },
  actions: { flexDirection: 'row', gap: 12, marginTop: 34, flexWrap: 'wrap' },
  primary: { minHeight: 53, flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: '#C8FF72', borderRadius: 15, paddingHorizontal: 19 },
  primaryText: { color: '#0A120D', fontSize: 15, fontWeight: '800', flexGrow: 1 },
  secondary: { minHeight: 53, flexDirection: 'row', alignItems: 'center', gap: 9, borderWidth: 1, borderColor: '#355147', borderRadius: 15, paddingHorizontal: 18, backgroundColor: 'rgba(20,40,30,.45)' },
  secondaryText: { color: '#DDE9E2', fontSize: 15, fontWeight: '700' },
  downloadSection: { marginBottom: 32, padding: Platform.OS === 'web' ? 28 : 20, borderWidth: 1, borderColor: '#365448', borderRadius: 26, backgroundColor: '#0B1914' },
  downloadHeading: { flexDirection: Platform.OS === 'web' ? 'row' : 'column', alignItems: Platform.OS === 'web' ? 'flex-end' : 'flex-start', justifyContent: 'space-between', gap: 18 },
  downloadTitle: { color: '#F0F7F1', fontSize: Platform.OS === 'web' ? 32 : 28, lineHeight: Platform.OS === 'web' ? 38 : 34, fontWeight: '800', letterSpacing: -1, marginTop: 10 },
  downloadLead: { color: '#93AA9F', fontSize: 14, lineHeight: 21, maxWidth: 390 },
  downloadGrid: { flexDirection: Platform.OS === 'web' ? 'row' : 'column', flexWrap: 'wrap', gap: 10, marginTop: 24 },
  downloadCard: { flexGrow: 1, flexBasis: Platform.OS === 'web' ? 240 : undefined, minHeight: 76, flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 15, borderWidth: 1, borderColor: '#315043', borderRadius: 17, backgroundColor: '#10231B' },
  downloadIcon: { width: 42, height: 42, borderRadius: 13, alignItems: 'center', justifyContent: 'center', backgroundColor: '#173226' },
  downloadCopy: { flex: 1 },
  downloadLabel: { color: '#EFF6F0', fontSize: 15, fontWeight: '800' },
  downloadDetail: { color: '#829A8E', fontSize: 11, marginTop: 3 },
  downloadNote: { color: '#71887D', fontSize: 10, lineHeight: 15, marginTop: 15 },
  briefing: { borderWidth: 1, borderColor: '#365448', borderRadius: 28, overflow: 'hidden', backgroundColor: '#0B1914' },
  briefingHeader: { minHeight: 54, paddingHorizontal: 21, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', borderBottomWidth: 1, borderBottomColor: '#294237', backgroundColor: 'rgba(17,40,29,.64)' },
  briefingLabel: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#FF7F6D' },
  briefingLabelText: { color: '#DCEAE3', fontSize: 10, fontWeight: '800', letterSpacing: 1.4 },
  briefingTime: { color: '#81988D', fontSize: 10, fontWeight: '800', letterSpacing: 1.05 },
  briefingBody: { flexDirection: Platform.OS === 'web' ? 'row' : 'column', minHeight: 382 },
  scenePane: { flex: 1.1, minHeight: 330, overflow: 'hidden', backgroundColor: '#10281D', padding: 20, position: 'relative' },
  sceneGrid: { position: 'absolute', top: 0, right: 0, bottom: 0, left: 0, opacity: 0.26, backgroundColor: '#133626', borderWidth: 1, borderColor: '#527160' },
  scanLine: { position: 'absolute', top: '47%', left: 0, right: 0, height: 1, backgroundColor: 'rgba(200,255,114,.58)' },
  sceneTopline: { flexDirection: 'row', justifyContent: 'space-between' },
  frameLabel: { color: '#B6CCBF', fontSize: 10, fontWeight: '800', letterSpacing: 1.1 },
  objectDoor: { position: 'absolute', right: '15%', top: '24%', width: '27%', height: '53%', borderRadius: 10, borderWidth: 1.5, borderColor: '#C8FF72', backgroundColor: 'rgba(200,255,114,.055)' },
  objectChair: { position: 'absolute', left: '18%', top: '48%', width: '21%', height: '26%', borderRadius: 10, borderWidth: 1.5, borderColor: '#C8FF72', backgroundColor: 'rgba(200,255,114,.055)' },
  objectTag: { position: 'absolute', top: -27, left: -2, backgroundColor: '#C8FF72', borderRadius: 5, paddingHorizontal: 7, paddingVertical: 4 },
  objectTagText: { color: '#102015', fontSize: 9, fontWeight: '800', letterSpacing: 0.35 },
  sceneFooter: { position: 'absolute', left: 20, bottom: 20, flexDirection: 'row', alignItems: 'center', gap: 7, paddingHorizontal: 10, paddingVertical: 8, borderRadius: 99, backgroundColor: 'rgba(8,22,16,.75)' },
  sceneFooterText: { color: '#C8FF72', fontSize: 9, fontWeight: '800', letterSpacing: 1 },
  responsePane: { flex: 0.9, minHeight: 330, padding: 30, justifyContent: 'center', backgroundColor: '#0F2019' },
  responseEyebrow: { color: '#C8FF72', fontSize: 10, fontWeight: '800', letterSpacing: 1.4 },
  responseTitle: { color: '#F0F7F1', fontSize: 29, lineHeight: 34, fontWeight: '800', letterSpacing: -0.8, marginTop: 14, maxWidth: 340 },
  responseText: { color: '#A9BDB3', fontSize: 15, lineHeight: 23, marginTop: 13, maxWidth: 360 },
  waveRow: { flexDirection: 'row', alignItems: 'center', gap: 5, height: 58, marginTop: 24 },
  waveBar: { width: 4, borderRadius: 99, backgroundColor: '#C8FF72' },
  responseMeta: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 20, paddingTop: 16, borderTopWidth: 1, borderTopColor: '#2B4339' },
  metaChip: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  metaText: { color: '#A9C2B3', fontSize: 9, fontWeight: '800', letterSpacing: 1 },
  metaTime: { color: '#71867C', fontSize: 11, fontWeight: '700' },
  introSection: { marginTop: 104, maxWidth: 710 },
  sectionOverline: { color: '#C8FF72', fontSize: 10, fontWeight: '800', letterSpacing: 1.75 },
  sectionTitle: { color: '#F0F6F1', fontSize: Platform.OS === 'web' ? 47 : 38, lineHeight: Platform.OS === 'web' ? 51 : 42, fontWeight: '800', letterSpacing: -1.8, marginTop: 15, maxWidth: 660 },
  sectionLead: { color: '#91A99E', fontSize: 17, lineHeight: 26, maxWidth: 610, marginTop: 17 },
  cards: { flexDirection: Platform.OS === 'web' ? 'row' : 'column', gap: 14, marginTop: 34 },
  cardWrap: { flex: 1 },
  card: { flex: 1, minHeight: 240, padding: 24, borderWidth: 1, borderColor: '#2C493D', borderRadius: 22, backgroundColor: '#0C1B16' },
  cardTopline: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  cardIcon: { width: 45, height: 45, borderRadius: 14, backgroundColor: '#173226', alignItems: 'center', justifyContent: 'center' },
  cardEyebrow: { color: '#7F988C', fontSize: 10, fontWeight: '800', letterSpacing: 1.45 },
  cardTitle: { color: '#EAF3EC', fontSize: 21, lineHeight: 26, fontWeight: '800', letterSpacing: -0.45, marginTop: 30 },
  cardText: { color: '#91A89E', fontSize: 14, lineHeight: 21, marginTop: 11 },
  mapSection: {
    marginTop: 110, flexDirection: Platform.OS === 'web' ? 'row' : 'column', flexWrap: 'wrap', gap: 32, alignItems: 'center',
  },
  mapCopy: { flex: 1, minWidth: 280, maxWidth: 460 },
  mapTitle: { color: '#F0F6F1', fontSize: Platform.OS === 'web' ? 40 : 32, lineHeight: Platform.OS === 'web' ? 44 : 36, fontWeight: '800', letterSpacing: -1.4, marginTop: 15 },
  mapLead: { color: '#91A99E', fontSize: 16, lineHeight: 25, marginTop: 15 },
  mapPoints: { marginTop: 24, gap: 16 },
  mapPoint: { flexDirection: 'row', gap: 12, alignItems: 'flex-start' },
  mapPointIcon: { width: 34, height: 34, borderRadius: 11, backgroundColor: '#0F2230', borderWidth: 1, borderColor: '#24485C', alignItems: 'center', justifyContent: 'center' },
  mapPointCopy: { flex: 1, gap: 3 },
  mapPointTitle: { color: '#EAF3EE', fontSize: 14, fontWeight: '800' },
  mapPointText: { color: '#8EA59A', fontSize: 13, lineHeight: 19 },
  mapCta: { flexDirection: 'row', alignItems: 'center', gap: 8, alignSelf: 'flex-start', backgroundColor: '#7EC8FF', borderRadius: 12, paddingHorizontal: 16, paddingVertical: 11, marginTop: 26 },
  mapCtaText: { color: '#07110F', fontSize: 14, fontWeight: '800' },
  mapFrame: { flex: 1.35, minWidth: 300 },
  architecture: { marginTop: 110, padding: Platform.OS === 'web' ? 38 : 24, borderWidth: 1, borderColor: '#315043', borderRadius: 28, backgroundColor: '#0B1914' },
  architectureHeading: { flexDirection: Platform.OS === 'web' ? 'row' : 'column', justifyContent: 'space-between', gap: 24 },
  architectureTitle: { color: '#F0F7F1', fontSize: Platform.OS === 'web' ? 41 : 33, lineHeight: Platform.OS === 'web' ? 45 : 38, fontWeight: '800', letterSpacing: -1.5, marginTop: 14, maxWidth: 570 },
  architectureSummary: { color: '#93AA9F', fontSize: 15, lineHeight: 23, maxWidth: 300, paddingTop: Platform.OS === 'web' ? 24 : 0 },
  pipeline: { flexDirection: Platform.OS === 'web' ? 'row' : 'column', marginTop: 42, alignItems: Platform.OS === 'web' ? 'stretch' : 'center' },
  pipelineItemWrap: { flex: 1, flexDirection: Platform.OS === 'web' ? 'row' : 'column', alignItems: 'stretch' },
  pipelineItem: { flex: 1, minHeight: 282, padding: 20, borderWidth: 1, borderColor: '#2C493D', borderRadius: 18, backgroundColor: '#10231B' },
  pipelineTopline: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  pipelineStep: { color: '#C8FF72', fontSize: 11, fontWeight: '800', letterSpacing: 1.2 },
  pipelineLabel: { color: '#829A8E', fontSize: 10, fontWeight: '800', letterSpacing: 1.25, marginTop: 32 },
  pipelineTitle: { color: '#EFF6F0', fontSize: 20, lineHeight: 24, fontWeight: '800', marginTop: 8, letterSpacing: -0.4 },
  pipelineText: { color: '#90A79C', fontSize: 13, lineHeight: 20, marginTop: 10 },
  pipelineMeta: { alignSelf: 'flex-start', borderRadius: 99, borderWidth: 1, borderColor: '#3C5A4B', paddingHorizontal: 9, paddingVertical: 6, marginTop: 19 },
  pipelineMetaText: { color: '#B5CCBF', fontSize: 9, fontWeight: '800', letterSpacing: 0.65 },
  connector: { width: Platform.OS === 'web' ? 46 : undefined, height: Platform.OS === 'web' ? undefined : 39, alignItems: 'center', justifyContent: 'center', flexDirection: Platform.OS === 'web' ? 'row' : 'column' },
  connectorLine: { width: Platform.OS === 'web' ? 18 : 1, height: Platform.OS === 'web' ? 1 : 18, backgroundColor: '#567164' },
  architectureFooter: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, padding: 15, marginTop: 28, borderRadius: 14, backgroundColor: '#102219' },
  architectureFooterText: { flex: 1, color: '#9DB4A9', fontSize: 12, lineHeight: 18 },
  safety: { flexDirection: 'row', alignItems: 'flex-start', gap: 14, maxWidth: 780, marginTop: 58, paddingVertical: 21, borderTopWidth: 1, borderBottomWidth: 1, borderColor: '#2C453A' },
  safetyIcon: { width: 38, height: 38, borderRadius: 12, backgroundColor: '#26352C', alignItems: 'center', justifyContent: 'center' },
  safetyCopy: { flex: 1 },
  safetyTitle: { color: '#E6EFE8', fontSize: 16, fontWeight: '800' },
  safetyText: { color: '#849B90', fontSize: 13, lineHeight: 20, marginTop: 5 },
  footer: { color: '#587065', fontSize: 10, fontWeight: '700', letterSpacing: 1.2, textAlign: 'center', marginTop: 54 },
});

const isElectron =
  typeof window !== 'undefined' &&
  (navigator.userAgent.toLowerCase().includes('electron') ||
    Boolean((window as any).process?.versions?.electron));

export default function IndexScreen() {
  const [forceHub, setForceHub] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const search = window.location.search;
      if (search.includes('view=hub') || search.includes('app=1')) {
        setForceHub(true);
      }
    }
  }, []);

  // Native Android/iOS or Electron desktop runs the Hub directly
  if (Platform.OS !== 'web' || isElectron || forceHub) {
    return <AssistantHub />;
  }

  return <LandingPage />;
}

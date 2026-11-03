import { Ionicons } from '@expo/vector-icons';
import { Link } from 'expo-router';
import { ReactNode, useEffect, useRef, useState } from 'react';
import {
  AccessibilityInfo,
  Animated,
  LayoutChangeEvent,
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
    icon: 'scan-outline' as const,
    eyebrow: 'CONTEXT',
    title: 'Know what is around you',
    text: 'Ask for a concise scene summary, readable text, or the location of an object in view.',
  },
  {
    icon: 'volume-high-outline' as const,
    eyebrow: 'CLARITY',
    title: 'Hear the essential detail',
    text: 'GuideLens returns a short response designed to be useful when spoken aloud, not a wall of text.',
  },
  {
    icon: 'shield-checkmark-outline' as const,
    eyebrow: 'CONTROL',
    title: 'Choose what leaves the phone',
    text: 'A single still frame and task prompt are sent for analysis. Video is not retained by default.',
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

export default function Landing() {
  const scrollY = useRef(new Animated.Value(0)).current;
  const reducedMotion = useReducedMotion();

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
          <View style={s.statusPill} accessibilityLabel="Prototype status: ready for demo">
            <View style={s.statusDot} />
            <Text style={s.statusText}>DEMO READY</Text>
          </View>
        </View>

        <View style={s.hero}>
          <View style={s.eyebrow}>
            <View style={s.eyebrowLine} />
            <Text style={s.eyebrowText}>SFBU × NVIDIA DGX SPARK 2026</Text>
          </View>
          <Text style={s.title}>Visual help, delivered with clarity.</Text>
          <Text style={s.subtitle}>
            A voice-first companion for everyday visual questions: understand a scene, read a label, or ask about what is in front of you.
          </Text>
          <View style={s.actions}>
            <Link href="/camera" asChild>
              <Pressable accessibilityRole="button" accessibilityLabel="Open the live camera demo" style={s.primary}>
                <Ionicons name="camera-outline" size={20} color="#0A120D" />
                <Text style={s.primaryText}>Open live demo</Text>
                <Ionicons name="arrow-forward" size={17} color="#0A120D" />
              </Pressable>
            </Link>
            <Pressable
              accessibilityRole="link"
              accessibilityLabel="Open the project on GitHub"
              onPress={() => Platform.OS === 'web' && window.open('https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026', '_blank')}
              style={s.secondary}
            >
              <Ionicons name="logo-github" size={19} color="#DDE9E2" />
              <Text style={s.secondaryText}>View project</Text>
            </Pressable>
          </View>
        </View>

        <View style={s.briefing}>
          <View style={s.briefingHeader}>
            <View style={s.briefingLabel}>
              <View style={s.liveDot} />
              <Text style={s.briefingLabelText}>LIVE SCENE BRIEFING</Text>
            </View>
            <Text style={s.briefingTime}>JUST NOW</Text>
          </View>
          <View style={s.briefingBody}>
            <View style={s.scenePane}>
              <View style={s.sceneGrid} />
              <View style={s.scanLine} />
              <View style={s.sceneTopline}>
                <Text style={s.frameLabel}>CAMERA INPUT</Text>
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
                <Text style={s.sceneFooterText}>SCENE RECOGNISED</Text>
              </View>
            </View>
            <View style={s.responsePane}>
              <Text style={s.responseEyebrow}>SPOKEN RESPONSE</Text>
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
            <Text style={s.architectureFooterText}>The hackathon app currently uses a mocked inference response. The API contract is ready to point at a DGX Spark-hosted vision-language model.</Text>
          </View>
        </Reveal>

        <Reveal scrollY={scrollY} reducedMotion={reducedMotion} style={s.safety}>
          <View style={s.safetyIcon}><Ionicons name="shield-checkmark-outline" size={20} color="#DDE9E2" /></View>
          <View style={s.safetyCopy}>
            <Text style={s.safetyTitle}>Assistive, never authoritative.</Text>
            <Text style={s.safetyText}>GuideLens can be wrong and must not be used as the sole source for traffic, medication, emergencies, or mobility decisions.</Text>
          </View>
        </Reveal>

        <Text style={s.footer}>BUILT FOR THE SFBU NVIDIA DGX SPARK HACKATHON · 2026</Text>
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

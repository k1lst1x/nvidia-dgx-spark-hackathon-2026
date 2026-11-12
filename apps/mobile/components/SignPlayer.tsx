import { Ionicons } from '@expo/vector-icons';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  HAND_CONNECTIONS,
  HandPose,
  REST_POSE,
  interpolatePose,
} from '../src/aslSignPoses';
import type { ASLToken } from '../src/aslGlossTranslator';

export type SignPlayerProps = {
  tokens: ASLToken[];
  autoPlay?: boolean;
  onSignComplete?: () => void;
  height?: number;
};

// Color palette for anatomical finger segments (standard MediaPipe accessibility scheme)
const FINGER_COLORS = {
  thumb: '#FFB800',
  index: '#00E5FF',
  middle: '#00E676',
  ring: '#7C4DFF',
  pinky: '#FF4081',
  palm: 'rgba(255, 255, 255, 0.4)',
  joint: '#FFFFFF',
  secondaryHand: 'rgba(183, 255, 86, 0.65)',
};

function getBoneColor(startIdx: number, endIdx: number): string {
  if (endIdx <= 4) return FINGER_COLORS.thumb;
  if (endIdx <= 8) return FINGER_COLORS.index;
  if (endIdx <= 12) return FINGER_COLORS.middle;
  if (endIdx <= 16) return FINGER_COLORS.ring;
  if (endIdx <= 20) return FINGER_COLORS.pinky;
  return FINGER_COLORS.palm;
}

export default function SignPlayer({
  tokens,
  autoPlay = true,
  onSignComplete,
  height = 320,
}: SignPlayerProps) {
  const [tokenIndex, setTokenIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playbackSpeed, setPlaybackSpeed] = useState<0.5 | 1.0 | 1.5>(1.0);
  const [isLooping, setIsLooping] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const stateRef = useRef({
    currentPose: REST_POSE,
    targetPose: REST_POSE,
    currentSecondary: undefined as HandPose | undefined,
    targetSecondary: undefined as HandPose | undefined,
    keyframeIdx: 0,
    startTime: 0,
    keyframeDuration: 400,
  });

  const activeToken = tokens[tokenIndex];

  // Draw hand skeleton onto canvas
  const drawHand = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      pose: HandPose,
      w: number,
      h: number,
      isSecondary = false,
    ) => {
      // Connect bones
      ctx.lineWidth = isSecondary ? 2.5 : 3.8;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = pose[start];
        const p2 = pose[end];
        if (!p1 || !p2) continue;

        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.strokeStyle = isSecondary ? FINGER_COLORS.secondaryHand : getBoneColor(start, end);
        ctx.stroke();
      }

      // Draw knuckles / joint dots
      for (let i = 0; i < pose.length; i++) {
        const pt = pose[i];
        if (!pt) continue;
        const x = pt.x * w;
        const y = pt.y * h;

        ctx.beginPath();
        const radius = i === 0 ? 5 : i === 4 || i === 8 || i === 12 || i === 16 || i === 20 ? 4.5 : 3;
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isSecondary ? '#B7FF56' : FINGER_COLORS.joint;
        ctx.fill();

        // Subtle glow on fingertips
        if (!isSecondary && (i === 4 || i === 8 || i === 12 || i === 16 || i === 20)) {
          ctx.beginPath();
          ctx.arc(x, y, radius + 2.5, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    },
    [],
  );

  // Main animation loop
  useEffect(() => {
    if (!tokens.length || !activeToken) return;

    const keyframes = activeToken.sign.keyframes;
    if (!keyframes || !keyframes.length) return;

    stateRef.current.keyframeIdx = 0;
    stateRef.current.startTime = performance.now();
    stateRef.current.currentPose = stateRef.current.targetPose || REST_POSE;
    stateRef.current.targetPose = keyframes[0].landmarks;
    stateRef.current.currentSecondary = stateRef.current.targetSecondary;
    stateRef.current.targetSecondary = keyframes[0].secondaryLandmarks;
    stateRef.current.keyframeDuration = (keyframes[0].durationMs || 400) / playbackSpeed;

    const render = (now: number) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      const displayW = canvas.clientWidth || 320;
      const displayH = canvas.clientHeight || height;

      if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
        canvas.width = displayW * dpr;
        canvas.height = displayH * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, displayW, displayH);

      // Background grid / visual studio depth
      ctx.fillStyle = '#081412';
      ctx.fillRect(0, 0, displayW, displayH);

      // Interpolation progress
      const elapsed = now - stateRef.current.startTime;
      const duration = stateRef.current.keyframeDuration;
      const t = Math.min(1, Math.max(0, elapsed / duration));

      const activePose = interpolatePose(stateRef.current.currentPose, stateRef.current.targetPose, t);

      // Draw secondary hand first if two-handed sign
      if (stateRef.current.targetSecondary) {
        const secStart = stateRef.current.currentSecondary || stateRef.current.targetSecondary;
        const activeSec = interpolatePose(secStart, stateRef.current.targetSecondary, t);
        drawHand(ctx, activeSec, displayW, displayH, true);
      }

      // Draw dominant hand
      drawHand(ctx, activePose, displayW, displayH, false);

      ctx.restore();

      // Check if current keyframe finished
      if (isPlaying && t >= 1) {
        const nextKeyframeIdx = stateRef.current.keyframeIdx + 1;
        if (nextKeyframeIdx < keyframes.length) {
          // Advance to next keyframe of the current sign
          stateRef.current.keyframeIdx = nextKeyframeIdx;
          stateRef.current.currentPose = stateRef.current.targetPose;
          stateRef.current.targetPose = keyframes[nextKeyframeIdx].landmarks;
          stateRef.current.currentSecondary = stateRef.current.targetSecondary;
          stateRef.current.targetSecondary = keyframes[nextKeyframeIdx].secondaryLandmarks;
          stateRef.current.keyframeDuration = (keyframes[nextKeyframeIdx].durationMs || 400) / playbackSpeed;
          stateRef.current.startTime = now;
        } else {
          // Advance to next word/token
          const nextTokenIdx = tokenIndex + 1;
          if (nextTokenIdx < tokens.length) {
            setTokenIndex(nextTokenIdx);
          } else {
            // Sequence completed
            onSignComplete?.();
            if (isLooping) {
              setTokenIndex(0);
            } else {
              setIsPlaying(false);
            }
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [
    tokens,
    tokenIndex,
    activeToken,
    isPlaying,
    playbackSpeed,
    isLooping,
    onSignComplete,
    drawHand,
    height,
  ]);

  const handlePrev = () => {
    setTokenIndex((curr) => Math.max(0, curr - 1));
  };

  const handleNext = () => {
    setTokenIndex((curr) => Math.min(tokens.length - 1, curr + 1));
  };

  const toggleSpeed = () => {
    setPlaybackSpeed((curr) => (curr === 0.5 ? 1.0 : curr === 1.0 ? 1.5 : 0.5));
  };

  if (!tokens.length) {
    return (
      <View style={[styles.emptyContainer, { height }]}>
        <Ionicons name="hand-right-outline" size={36} color="#4E6760" />
        <Text style={styles.emptyText}>Enter text or speak to generate sign translation</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 2D/3D Animated Hand Landmark Canvas */}
      <View style={[styles.canvasWrapper, { height }]}>
        {Platform.OS === 'web' ? (
          <canvas
            ref={canvasRef as any}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
        ) : (
          <View style={styles.fallbackRig}>
            <Ionicons name="hand-right" size={64} color="#B7FF56" />
          </View>
        )}

        {/* Current Gloss Banner Floating Over Canvas */}
        <View style={styles.glossOverlay}>
          <View style={styles.glossBadge}>
            <Text style={styles.glossBadgeText}>
              {activeToken.type === 'letter'
                ? 'FINGERSPELLING'
                : activeToken.type === 'number'
                ? 'NUMBER'
                : 'ASL SIGN'}
            </Text>
          </View>
          <Text style={styles.glossTitle}>{activeToken.gloss}</Text>
          <Text style={styles.glossDesc} numberOfLines={2}>
            {activeToken.sign.description}
          </Text>
        </View>

        {/* Playback Controls Overlay Bar */}
        <View style={styles.controlsBar}>
          <Pressable
            onPress={toggleSpeed}
            style={styles.speedBtn}
            accessibilityLabel={`Playback speed ${playbackSpeed}x`}
          >
            <Text style={styles.speedBtnText}>{playbackSpeed}x</Text>
          </Pressable>

          <View style={styles.playbackButtons}>
            <Pressable
              onPress={handlePrev}
              disabled={tokenIndex === 0}
              style={[styles.iconBtn, tokenIndex === 0 && styles.disabledBtn]}
              accessibilityLabel="Previous sign"
            >
              <Ionicons name="play-skip-back" size={18} color={tokenIndex === 0 ? '#4E6760' : '#FFFFFF'} />
            </Pressable>

            <Pressable
              onPress={() => setIsPlaying((p) => !p)}
              style={styles.playPauseBtn}
              accessibilityLabel={isPlaying ? 'Pause sign playback' : 'Play sign playback'}
            >
              <Ionicons name={isPlaying ? 'pause' : 'play'} size={22} color="#07110F" />
            </Pressable>

            <Pressable
              onPress={handleNext}
              disabled={tokenIndex === tokens.length - 1}
              style={[styles.iconBtn, tokenIndex === tokens.length - 1 && styles.disabledBtn]}
              accessibilityLabel="Next sign"
            >
              <Ionicons
                name="play-skip-forward"
                size={18}
                color={tokenIndex === tokens.length - 1 ? '#4E6760' : '#FFFFFF'}
              />
            </Pressable>
          </View>

          <Pressable
            onPress={() => setIsLooping((l) => !l)}
            style={[styles.loopBtn, isLooping && styles.loopBtnActive]}
            accessibilityLabel={isLooping ? 'Looping enabled' : 'Looping disabled'}
          >
            <Ionicons name="repeat" size={16} color={isLooping ? '#B7FF56' : '#8CA299'} />
          </Pressable>
        </View>
      </View>

      {/* Horizontal Word/Token Chips (Interactive scrubber) */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.chipsContainer}
      >
        {tokens.map((token, index) => {
          const isActive = index === tokenIndex;
          return (
            <Pressable
              key={token.id}
              onPress={() => {
                setTokenIndex(index);
                setIsPlaying(true);
              }}
              style={[
                styles.tokenChip,
                isActive && styles.tokenChipActive,
                token.type === 'letter' && styles.letterChip,
              ]}
              accessibilityLabel={`Sign ${token.gloss}`}
            >
              <Text style={[styles.tokenGloss, isActive && styles.tokenGlossActive]}>
                {token.gloss}
              </Text>
              {token.type === 'word' && (
                <Text style={[styles.tokenSubtext, isActive && styles.tokenSubtextActive]}>
                  {token.text}
                </Text>
              )}
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    backgroundColor: '#07110F',
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  canvasWrapper: {
    width: '100%',
    position: 'relative',
    backgroundColor: '#081412',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fallbackRig: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  glossOverlay: {
    position: 'absolute',
    top: 12,
    left: 14,
    right: 14,
    pointerEvents: 'none',
  },
  glossBadge: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(183, 255, 86, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.4)',
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 2,
    marginBottom: 4,
  },
  glossBadgeText: {
    color: '#B7FF56',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.6,
  },
  glossTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '900',
    letterSpacing: 0.5,
    textShadowColor: 'rgba(0, 0, 0, 0.8)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  glossDesc: {
    color: '#B8CCC5',
    fontSize: 12,
    marginTop: 2,
    lineHeight: 16,
    textShadowColor: 'rgba(0, 0, 0, 0.8)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  controlsBar: {
    position: 'absolute',
    bottom: 12,
    left: 14,
    right: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(7, 17, 15, 0.82)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  speedBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  speedBtnText: {
    color: '#DCEAE4',
    fontSize: 11,
    fontWeight: '700',
  },
  playbackButtons: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  iconBtn: {
    padding: 6,
  },
  disabledBtn: {
    opacity: 0.35,
  },
  playPauseBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#B7FF56',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loopBtn: {
    padding: 6,
    borderRadius: 6,
  },
  loopBtnActive: {
    backgroundColor: 'rgba(183, 255, 86, 0.15)',
  },
  chipsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.25)',
  },
  tokenChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
    alignItems: 'center',
  },
  tokenChipActive: {
    backgroundColor: 'rgba(183, 255, 86, 0.2)',
    borderColor: '#B7FF56',
  },
  letterChip: {
    minWidth: 32,
    paddingHorizontal: 8,
  },
  tokenGloss: {
    color: '#DCEAE4',
    fontSize: 13,
    fontWeight: '800',
  },
  tokenGlossActive: {
    color: '#B7FF56',
  },
  tokenSubtext: {
    color: '#8CA299',
    fontSize: 9,
    marginTop: 1,
  },
  tokenSubtextActive: {
    color: '#DCEAE4',
  },
  emptyContainer: {
    width: '100%',
    backgroundColor: '#081412',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    padding: 24,
  },
  emptyText: {
    color: '#8CA299',
    fontSize: 13,
    marginTop: 10,
    textAlign: 'center',
  },
});

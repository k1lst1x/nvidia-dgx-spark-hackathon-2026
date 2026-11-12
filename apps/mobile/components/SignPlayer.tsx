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
  initialMode?: 'avatar' | 'skeleton';
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

// Natural human skin, hair, and clothing palette
const SKIN = {
  base: '#ECC09E',
  shadow: '#CFA07E',
  highlight: '#FCE4D0',
  blush: 'rgba(224, 112, 112, 0.18)',
  lip: '#C27474',
  nail: '#F5D3BD',
  nailShine: '#FFFFFF',
  hair: '#22201F',
  hairHighlight: '#3B3734',
  hoodie: '#10261E',
  hoodieShadow: '#091612',
  hoodieTrim: '#1E4738',
  badge: '#0D201A',
  accent: '#B7FF56',
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
  height = 340,
  initialMode = 'avatar',
}: SignPlayerProps) {
  const [tokenIndex, setTokenIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playbackSpeed, setPlaybackSpeed] = useState<0.5 | 1.0 | 1.5>(1.0);
  const [isLooping, setIsLooping] = useState(true);
  const [viewMode, setViewMode] = useState<'avatar' | 'skeleton'>(initialMode);

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

  // Draw technical 21-point MediaPipe skeleton
  const drawSkeletonHand = useCallback(
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

  // Draw volumetric human hand with skin tone, organic palm & finger contours and fingernails
  const drawVolumetricHand = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      pose: HandPose,
      w: number,
      h: number,
      isSecondary = false,
    ) => {
      if (!pose || !pose.length) return;

      const wrist = pose[0];
      const thumbCmc = pose[1];
      const indexMcp = pose[5];
      const middleMcp = pose[9];
      const ringMcp = pose[13];
      const pinkyMcp = pose[17];

      // 1. Palm base fill
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(wrist.x * w, wrist.y * h);
      ctx.lineTo(thumbCmc.x * w, thumbCmc.y * h);
      ctx.lineTo(indexMcp.x * w, indexMcp.y * h);
      ctx.lineTo(middleMcp.x * w, middleMcp.y * h);
      ctx.lineTo(ringMcp.x * w, ringMcp.y * h);
      ctx.lineTo(pinkyMcp.x * w, pinkyMcp.y * h);
      ctx.closePath();

      const palmGrad = ctx.createLinearGradient(
        wrist.x * w,
        wrist.y * h,
        middleMcp.x * w,
        middleMcp.y * h,
      );
      palmGrad.addColorStop(0, isSecondary ? '#BAA088' : SKIN.shadow);
      palmGrad.addColorStop(1, isSecondary ? '#DFCAA6' : SKIN.base);
      ctx.fillStyle = palmGrad;
      ctx.fill();

      // Soft palm outline & shadow
      ctx.strokeStyle = isSecondary ? 'rgba(90, 75, 60, 0.35)' : 'rgba(120, 80, 50, 0.4)';
      ctx.lineWidth = 1.8;
      ctx.stroke();

      // 2. Volumetric finger segments [MCP, PIP, DIP, TIP]
      const fingerChains = [
        [1, 2, 3, 4], // Thumb
        [5, 6, 7, 8], // Index
        [9, 10, 11, 12], // Middle
        [13, 14, 15, 16], // Ring
        [17, 18, 19, 20], // Pinky
      ];

      for (const chain of fingerChains) {
        for (let seg = 0; seg < chain.length - 1; seg++) {
          const p1 = pose[chain[seg]];
          const p2 = pose[chain[seg + 1]];
          if (!p1 || !p2) continue;

          const isTip = seg === chain.length - 2;
          const strokeWidth = seg === 0 ? 8.5 : seg === 1 ? 7.2 : 6.0;

          // Finger segment bone/flesh
          ctx.beginPath();
          ctx.moveTo(p1.x * w, p1.y * h);
          ctx.lineTo(p2.x * w, p2.y * h);
          ctx.lineWidth = isSecondary ? strokeWidth * 0.9 : strokeWidth;
          ctx.lineCap = 'round';
          ctx.strokeStyle = isSecondary ? '#D8C3A5' : SKIN.base;
          ctx.stroke();

          // Subtle joint highlight/crease
          ctx.beginPath();
          ctx.arc(p2.x * w, p2.y * h, strokeWidth * 0.4, 0, Math.PI * 2);
          ctx.fillStyle = isSecondary ? '#C0AB8E' : SKIN.highlight;
          ctx.fill();

          // Fingernail on tip
          if (isTip) {
            const tip = p2;
            const dip = p1;
            const dx = tip.x - dip.x;
            const dy = tip.y - dip.y;
            const angle = Math.atan2(dy, dx);

            ctx.save();
            ctx.translate(tip.x * w, tip.y * h);
            ctx.rotate(angle);
            ctx.beginPath();
            ctx.ellipse(0, 0, 3.2, 2.0, 0, 0, Math.PI * 2);
            ctx.fillStyle = SKIN.nail;
            ctx.fill();

            // Nail specular highlight
            ctx.beginPath();
            ctx.arc(0.8, -0.6, 0.8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.fill();
            ctx.restore();
          }
        }
      }

      ctx.restore();
    },
    [],
  );

  // Draw full 3D human avatar signer with torso, expressive face (NMM), organic arms and hands
  const drawHumanAvatar = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      activePose: HandPose,
      activeSecondary: HandPose | undefined,
      w: number,
      h: number,
      token: ASLToken,
      now: number,
    ) => {
      // 1. Studio backdrop lighting
      const bgGrad = ctx.createRadialGradient(
        w * 0.5,
        h * 0.35,
        15,
        w * 0.5,
        h * 0.4,
        Math.max(w, h) * 0.75,
      );
      bgGrad.addColorStop(0, '#122D24');
      bgGrad.addColorStop(0.55, '#0A1814');
      bgGrad.addColorStop(1, '#06100D');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Subtle atmospheric grid floor
      ctx.strokeStyle = 'rgba(183, 255, 86, 0.04)';
      ctx.lineWidth = 1;
      for (let gy = h * 0.75; gy < h; gy += 18) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
      }

      // Proportions & subtle breathing
      const breath = Math.sin(now / 1300) * 1.6;
      const cx = w * 0.5;
      const headY = h * 0.22 + breath * 0.4;
      const hrx = Math.min(w * 0.11, h * 0.135);
      const hry = hrx * 1.28;

      const lsx = cx - hrx * 2.6;
      const lsy = h * 0.45 + breath;
      const rsx = cx + hrx * 2.6;
      const rsy = h * 0.45 + breath;

      // 2. Torso & Tech Hoodie
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(lsx, lsy);
      ctx.bezierCurveTo(cx - hrx * 1.6, lsy - 4, cx - hrx * 0.8, h * 0.44, cx - hrx * 0.4, h * 0.45);
      ctx.lineTo(cx, h * 0.465); // collar notch
      ctx.lineTo(cx + hrx * 0.4, h * 0.45);
      ctx.bezierCurveTo(cx + hrx * 0.8, h * 0.44, cx + hrx * 1.6, lsy - 4, rsx, rsy);
      ctx.lineTo(w * 0.82, h);
      ctx.lineTo(w * 0.18, h);
      ctx.closePath();

      const hoodieGrad = ctx.createLinearGradient(cx, h * 0.42, cx, h);
      hoodieGrad.addColorStop(0, SKIN.hoodie);
      hoodieGrad.addColorStop(1, SKIN.hoodieShadow);
      ctx.fillStyle = hoodieGrad;
      ctx.fill();

      // Hoodie zipper line & collar trim
      ctx.beginPath();
      ctx.moveTo(cx, h * 0.465);
      ctx.lineTo(cx, h);
      ctx.strokeStyle = SKIN.hoodieTrim;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // SFBU ASL Emblem Badge on left chest
      const badgeX = cx - hrx * 1.45;
      const badgeY = h * 0.56 + breath;
      ctx.beginPath();
      ctx.roundRect(badgeX - 16, badgeY - 7, 32, 14, 4);
      ctx.fillStyle = SKIN.badge;
      ctx.fill();
      ctx.strokeStyle = 'rgba(183, 255, 86, 0.45)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = SKIN.accent;
      ctx.font = 'bold 7px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('SFBU ASL', badgeX, badgeY);

      // 3. Neck
      ctx.beginPath();
      ctx.moveTo(cx - hrx * 0.35, headY + hry * 0.65);
      ctx.lineTo(cx - hrx * 0.35, h * 0.46);
      ctx.lineTo(cx + hrx * 0.35, h * 0.46);
      ctx.lineTo(cx + hrx * 0.35, headY + hry * 0.65);
      ctx.closePath();

      const neckGrad = ctx.createLinearGradient(cx, headY + hry * 0.65, cx, h * 0.46);
      neckGrad.addColorStop(0, SKIN.shadow);
      neckGrad.addColorStop(1, SKIN.base);
      ctx.fillStyle = neckGrad;
      ctx.fill();

      // 4. Head Oval & Face
      ctx.beginPath();
      ctx.ellipse(cx, headY, hrx, hry, 0, 0, Math.PI * 2);
      const faceGrad = ctx.createRadialGradient(
        cx - hrx * 0.25,
        headY - hry * 0.2,
        4,
        cx,
        headY,
        hrx * 1.1,
      );
      faceGrad.addColorStop(0, SKIN.highlight);
      faceGrad.addColorStop(0.7, SKIN.base);
      faceGrad.addColorStop(1, SKIN.shadow);
      ctx.fillStyle = faceGrad;
      ctx.fill();

      // Ears
      ctx.beginPath();
      ctx.ellipse(cx - hrx * 0.98, headY + hry * 0.05, hrx * 0.16, hry * 0.26, 0, 0, Math.PI * 2);
      ctx.ellipse(cx + hrx * 0.98, headY + hry * 0.05, hrx * 0.16, hry * 0.26, 0, 0, Math.PI * 2);
      ctx.fillStyle = SKIN.base;
      ctx.fill();

      // Hair (stylish modern cropped haircut)
      ctx.beginPath();
      ctx.ellipse(cx, headY - hry * 0.45, hrx * 1.04, hry * 0.68, 0, Math.PI, Math.PI * 2);
      ctx.lineTo(cx + hrx * 1.02, headY - hry * 0.05);
      ctx.bezierCurveTo(
        cx + hrx * 0.8,
        headY - hry * 0.5,
        cx - hrx * 0.2,
        headY - hry * 0.4,
        cx - hrx * 1.02,
        headY - hry * 0.05,
      );
      ctx.closePath();

      const hairGrad = ctx.createLinearGradient(cx, headY - hry * 1.1, cx, headY);
      hairGrad.addColorStop(0, SKIN.hairHighlight);
      hairGrad.addColorStop(1, SKIN.hair);
      ctx.fillStyle = hairGrad;
      ctx.fill();

      // 5. Facial Features & Linguistic Non-Manual Markers (NMM)
      const isBlinking = now % 3800 < 130;
      const isWhQuestion = /WHERE|WHAT|WHO|WHY|WHEN|HOW/.test(token?.gloss || '');
      const isAffirmative = /HELLO|THANK|WELCOME|LOVE|YES|GOOD|PLEASE/.test(token?.gloss || '');

      const eyeY = headY - hry * 0.06;
      const leftEyeX = cx - hrx * 0.36;
      const rightEyeX = cx + hrx * 0.36;

      if (isBlinking) {
        // Closed eyelids
        ctx.strokeStyle = '#382516';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(leftEyeX, eyeY, hrx * 0.16, 0.1, Math.PI - 0.1);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(rightEyeX, eyeY, hrx * 0.16, 0.1, Math.PI - 0.1);
        ctx.stroke();
      } else {
        // Open eyes (sclera + amber iris + pupil + specular shine)
        for (const ex of [leftEyeX, rightEyeX]) {
          ctx.beginPath();
          ctx.ellipse(ex, eyeY, hrx * 0.18, hry * 0.11, 0, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();
          ctx.strokeStyle = 'rgba(70, 45, 25, 0.35)';
          ctx.lineWidth = 1;
          ctx.stroke();

          // Iris
          ctx.beginPath();
          ctx.arc(ex, eyeY, hry * 0.08, 0, Math.PI * 2);
          ctx.fillStyle = '#4A301D';
          ctx.fill();

          // Pupil
          ctx.beginPath();
          ctx.arc(ex, eyeY, hry * 0.045, 0, Math.PI * 2);
          ctx.fillStyle = '#0F0B08';
          ctx.fill();

          // Specular catchlight (adds life)
          ctx.beginPath();
          ctx.arc(ex + 1.8, eyeY - 1.8, 1.2, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();
        }
      }

      // Eyebrows (Crucial ASL Grammar Marker)
      // WH-questions: furrowed downward towards center
      // Affirmative: arched slightly upwards
      // Neutral: relaxed standard arch
      const browY = eyeY - hry * 0.22;
      ctx.strokeStyle = '#27221E';
      ctx.lineWidth = 2.6;
      ctx.lineCap = 'round';

      // Left eyebrow
      ctx.beginPath();
      if (isWhQuestion) {
        // Furrowed down toward nose
        ctx.moveTo(leftEyeX - hrx * 0.18, browY - 1);
        ctx.lineTo(leftEyeX + hrx * 0.18, browY + 4);
      } else if (isAffirmative) {
        // Lifted arch
        ctx.moveTo(leftEyeX - hrx * 0.18, browY - 2);
        ctx.quadraticCurveTo(leftEyeX, browY - 7, leftEyeX + hrx * 0.18, browY - 2);
      } else {
        // Neutral
        ctx.moveTo(leftEyeX - hrx * 0.18, browY);
        ctx.quadraticCurveTo(leftEyeX, browY - 4, leftEyeX + hrx * 0.18, browY - 1);
      }
      ctx.stroke();

      // Right eyebrow
      ctx.beginPath();
      if (isWhQuestion) {
        // Furrowed down toward nose
        ctx.moveTo(rightEyeX - hrx * 0.18, browY + 4);
        ctx.lineTo(rightEyeX + hrx * 0.18, browY - 1);
      } else if (isAffirmative) {
        // Lifted arch
        ctx.moveTo(rightEyeX - hrx * 0.18, browY - 2);
        ctx.quadraticCurveTo(rightEyeX, browY - 7, rightEyeX + hrx * 0.18, browY - 2);
      } else {
        // Neutral
        ctx.moveTo(rightEyeX - hrx * 0.18, browY - 1);
        ctx.quadraticCurveTo(rightEyeX, browY - 4, rightEyeX + hrx * 0.18, browY);
      }
      ctx.stroke();

      // Nose
      const noseY = headY + hry * 0.15;
      ctx.beginPath();
      ctx.moveTo(cx, noseY - hry * 0.12);
      ctx.lineTo(cx + 2, noseY);
      ctx.lineTo(cx - 3, noseY + 2);
      ctx.strokeStyle = 'rgba(170, 115, 80, 0.45)';
      ctx.lineWidth = 1.4;
      ctx.stroke();

      // Mouth
      const mouthY = headY + hry * 0.44;
      ctx.beginPath();
      if (isAffirmative) {
        // Friendly smile
        ctx.arc(cx, mouthY - 3, hrx * 0.28, 0.2, Math.PI - 0.2);
        ctx.strokeStyle = SKIN.lip;
        ctx.lineWidth = 2.4;
        ctx.stroke();
      } else if (isWhQuestion) {
        // Slightly open question mouth
        ctx.ellipse(cx, mouthY, hrx * 0.14, hry * 0.08, 0, 0, Math.PI * 2);
        ctx.fillStyle = '#421E1E';
        ctx.fill();
        ctx.strokeStyle = SKIN.lip;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      } else {
        // Calm neutral lips
        ctx.moveTo(cx - hrx * 0.22, mouthY);
        ctx.quadraticCurveTo(cx, mouthY + 1.5, cx + hrx * 0.22, mouthY);
        ctx.strokeStyle = SKIN.lip;
        ctx.lineWidth = 2.2;
        ctx.stroke();
      }

      // 6. Connected Arms & Forearms
      // Right dominant arm (starts at rsx, rsy and connects to wrist activePose[0])
      const rwX = activePose[0].x * w;
      const rwY = activePose[0].y * h;
      const relbowX = Math.max(rsx + 10, (rsx + rwX) / 2 + w * 0.08);
      const relbowY = Math.max(rsy + 20, (rsy + rwY) / 2 + h * 0.09);

      // Upper arm
      ctx.beginPath();
      ctx.moveTo(rsx, rsy);
      ctx.lineTo(relbowX, relbowY);
      ctx.strokeStyle = SKIN.hoodie;
      ctx.lineWidth = 16;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Forearm connecting to wrist
      ctx.beginPath();
      ctx.moveTo(relbowX, relbowY);
      ctx.lineTo(rwX, rwY);
      ctx.strokeStyle = SKIN.hoodieTrim;
      ctx.lineWidth = 13;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Sleeve cuff
      ctx.beginPath();
      ctx.arc(rwX, rwY, 8, 0, Math.PI * 2);
      ctx.fillStyle = SKIN.hoodie;
      ctx.fill();

      // Left arm (Secondary hand or resting hand)
      if (activeSecondary) {
        const lwX = activeSecondary[0].x * w;
        const lwY = activeSecondary[0].y * h;
        const lelbowX = Math.min(lsx - 10, (lsx + lwX) / 2 - w * 0.08);
        const lelbowY = Math.max(lsy + 20, (lsy + lwY) / 2 + h * 0.09);

        // Left upper arm & forearm
        ctx.beginPath();
        ctx.moveTo(lsx, lsy);
        ctx.lineTo(lelbowX, lelbowY);
        ctx.strokeStyle = SKIN.hoodie;
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(lelbowX, lelbowY);
        ctx.lineTo(lwX, lwY);
        ctx.strokeStyle = SKIN.hoodieTrim;
        ctx.lineWidth = 12;
        ctx.lineCap = 'round';
        ctx.stroke();
      } else {
        // Natural resting left arm at side/hip
        const restElbowX = lsx - 12;
        const restElbowY = h * 0.65;
        const restHandX = cx - hrx * 1.5;
        const restHandY = h * 0.88;

        ctx.beginPath();
        ctx.moveTo(lsx, lsy);
        ctx.lineTo(restElbowX, restElbowY);
        ctx.lineTo(restHandX, restHandY);
        ctx.strokeStyle = SKIN.hoodie;
        ctx.lineWidth = 14;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();

        // Resting hand shape
        ctx.beginPath();
        ctx.ellipse(restHandX, restHandY, 11, 7, -0.4, 0, Math.PI * 2);
        ctx.fillStyle = SKIN.shadow;
        ctx.fill();
      }

      // 7. Depth sorted volumetric hands
      // Compare average z to see which hand is closer to camera
      const rightZ = activePose[0]?.z || 0;
      const leftZ = activeSecondary ? activeSecondary[0]?.z || 0 : 999;

      if (activeSecondary && leftZ < rightZ) {
        // Left hand is in front
        drawVolumetricHand(ctx, activePose, w, h, false);
        drawVolumetricHand(ctx, activeSecondary, w, h, true);
      } else {
        // Right hand is in front (default)
        if (activeSecondary) {
          drawVolumetricHand(ctx, activeSecondary, w, h, true);
        }
        drawVolumetricHand(ctx, activePose, w, h, false);
      }

      ctx.restore();
    },
    [drawVolumetricHand],
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
      const displayW = canvas.clientWidth || 340;
      const displayH = canvas.clientHeight || height;

      if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
        canvas.width = displayW * dpr;
        canvas.height = displayH * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, displayW, displayH);

      // Interpolation progress
      const elapsed = now - stateRef.current.startTime;
      const duration = stateRef.current.keyframeDuration;
      const t = Math.min(1, Math.max(0, elapsed / duration));

      const activePose = interpolatePose(stateRef.current.currentPose, stateRef.current.targetPose, t);
      const secStart = stateRef.current.currentSecondary || stateRef.current.targetSecondary;
      const activeSecondary = stateRef.current.targetSecondary
        ? interpolatePose(secStart!, stateRef.current.targetSecondary, t)
        : undefined;

      if (viewMode === 'avatar') {
        // 3D Human Avatar Renderer
        drawHumanAvatar(ctx, activePose, activeSecondary, displayW, displayH, activeToken, now);
      } else {
        // Technical 21-point MediaPipe X-Ray Skeleton Renderer
        ctx.fillStyle = '#081412';
        ctx.fillRect(0, 0, displayW, displayH);

        // Studio center guidelines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(displayW * 0.5, 0);
        ctx.lineTo(displayW * 0.5, displayH);
        ctx.moveTo(0, displayH * 0.5);
        ctx.lineTo(displayW, displayH * 0.5);
        ctx.stroke();

        if (activeSecondary) {
          drawSkeletonHand(ctx, activeSecondary, displayW, displayH, true);
        }
        drawSkeletonHand(ctx, activePose, displayW, displayH, false);
      }

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
    viewMode,
    onSignComplete,
    drawSkeletonHand,
    drawHumanAvatar,
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
        <Ionicons name="person-outline" size={36} color="#4E6760" />
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
            <Ionicons name="person" size={64} color="#B7FF56" />
          </View>
        )}

        {/* Current Gloss Banner Floating Over Canvas */}
        <View style={styles.glossOverlay}>
          <View style={styles.glossTopRow}>
            <View style={styles.glossBadge}>
              <Text style={styles.glossBadgeText}>
                {activeToken.type === 'letter'
                  ? 'FINGERSPELLING'
                  : activeToken.type === 'number'
                  ? 'NUMBER'
                  : 'ASL SIGN'}
              </Text>
            </View>

            {/* Avatar / Skeleton Switcher Pill */}
            <Pressable
              onPress={() => setViewMode((m) => (m === 'avatar' ? 'skeleton' : 'avatar'))}
              style={[styles.viewModeToggle, viewMode === 'avatar' && styles.viewModeToggleActive]}
              accessibilityRole="button"
              accessibilityLabel={`Switch to ${viewMode === 'avatar' ? 'X-Ray Skeleton' : '3D Human Avatar'} view`}
            >
              <Ionicons
                name={viewMode === 'avatar' ? 'person' : 'hardware-chip-outline'}
                size={12}
                color={viewMode === 'avatar' ? '#07110F' : '#B7FF56'}
              />
              <Text
                style={[
                  styles.viewModeToggleText,
                  viewMode === 'avatar' && styles.viewModeToggleTextActive,
                ]}
              >
                {viewMode === 'avatar' ? '3D Human Avatar' : 'X-Ray Skeleton'}
              </Text>
            </Pressable>
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
              <Text
                style={[
                  styles.tokenGloss,
                  isActive && styles.tokenGlossActive,
                ]}
              >
                {token.gloss}
              </Text>
              <Text
                style={[
                  styles.tokenSubtext,
                  isActive && styles.tokenSubtextActive,
                ]}
              >
                {token.text}
              </Text>
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
  },
  canvasWrapper: {
    width: '100%',
    position: 'relative',
    backgroundColor: '#07130F',
  },
  fallbackRig: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  glossOverlay: {
    position: 'absolute',
    top: 12,
    left: 14,
    right: 14,
    pointerEvents: 'box-none',
  },
  glossTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  glossBadge: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(183, 255, 86, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.4)',
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  glossBadgeText: {
    color: '#B7FF56',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.6,
  },
  viewModeToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(10, 24, 20, 0.75)',
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.35)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  viewModeToggleActive: {
    backgroundColor: '#B7FF56',
    borderColor: '#B7FF56',
  },
  viewModeToggleText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '700',
  },
  viewModeToggleTextActive: {
    color: '#07110F',
    fontWeight: '800',
  },
  glossTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '900',
    letterSpacing: 0.5,
    textShadowColor: 'rgba(0, 0, 0, 0.85)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  glossDesc: {
    color: '#B8CCC5',
    fontSize: 12,
    marginTop: 2,
    lineHeight: 16,
    textShadowColor: 'rgba(0, 0, 0, 0.85)',
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
    backgroundColor: 'rgba(7, 17, 15, 0.85)',
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

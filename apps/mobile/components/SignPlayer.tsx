import { Ionicons } from '@expo/vector-icons';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import * as THREE from 'three';
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
  initialMode?: '3d' | 'skeleton';
};

// Color palette for 2D MediaPipe skeleton
const FINGER_COLORS = {
  thumb: '#FFB800',
  index: '#00E5FF',
  middle: '#00E676',
  ring: '#7C4DFF',
  pinky: '#FF4081',
  palm: 'rgba(255, 255, 255, 0.4)',
  joint: '#FFFFFF',
  secondaryHand: 'rgba(183, 255, 86, 0.7)',
};

function getBoneColor(startIdx: number, endIdx: number): string {
  if (endIdx <= 4) return FINGER_COLORS.thumb;
  if (endIdx <= 8) return FINGER_COLORS.index;
  if (endIdx <= 12) return FINGER_COLORS.middle;
  if (endIdx <= 16) return FINGER_COLORS.ring;
  if (endIdx <= 20) return FINGER_COLORS.pinky;
  return FINGER_COLORS.palm;
}

// Handshape classification lookup for authentic ASL linguistic guidance
const ASL_HANDSHAPE_GUIDE: Record<
  string,
  {
    handshape: string;
    palmFacing: string;
    motionGuide: string;
    facialNMM: string;
  }
> = {
  HELLO: {
    handshape: 'ASL "Open-B" Handshape (Flat hand, thumb alongside)',
    palmFacing: 'Palm Forward / Outward (Facing recipient)',
    motionGuide: 'Begin with index edge near temple, push forward and outward 6–8" in a gentle saluting wave.',
    facialNMM: 'Warm pleasant eye contact with a gentle welcoming smile.',
  },
  'THANK-YOU': {
    handshape: 'ASL "Flat-B" Handshape (All fingers extended, touching)',
    palmFacing: 'Palm Inward initially, opening forward toward recipient',
    motionGuide: 'Fingertips gently touch chin/lips, then hand moves forward and slightly downward toward recipient.',
    facialNMM: 'Slight head bow and appreciative smile (Non-Manual Marker).',
  },
  PLEASE: {
    handshape: 'ASL "Flat-5" Handshape (Open palm, thumb relaxed)',
    palmFacing: 'Palm Inward (Touching / facing chest)',
    motionGuide: 'Flat open palm rubs circular clockwise motion over chest 2–3 times.',
    facialNMM: 'Humble pleasant facial expression with eyes open.',
  },
  HELP: {
    handshape: 'Dominant "Closed-A" / Fist over Non-dominant "Open-B" palm base',
    palmFacing: 'Dominant thumb pointing UP; Base palm facing UP',
    motionGuide: 'Rest closed fist upon open palm, lift both hands together upward 6–8" as a unified motion.',
    facialNMM: 'Attentive, helpful expression; furrowed brows if asking for help.',
  },
  YES: {
    handshape: 'ASL "S" Fist Handshape',
    palmFacing: 'Palm Forward / Outward',
    motionGuide: 'Fist nods at wrist forward and down 2–3 times, mimicking a nodding head.',
    facialNMM: 'Nod head in affirmative synchronization with fist nods.',
  },
  NO: {
    handshape: 'ASL "3" modified: Index + Middle snap shut against Thumb',
    palmFacing: 'Palm Forward / Outward',
    motionGuide: 'Index and middle fingers snap firmly against the thumb pad, like a closing mouth.',
    facialNMM: 'Firm slight head shake in negative sync.',
  },
  STOP: {
    handshape: 'Dominant "Open-B" chopping into Non-dominant "Open-B" palm',
    palmFacing: 'Dominant palm facing side; Base palm facing UP',
    motionGuide: 'Dominant pinky edge comes down firmly onto the flat upward base palm with decisive stop.',
    facialNMM: 'Direct firm gaze, neutral or cautionary expression.',
  },
  WAIT: {
    handshape: 'Both hands "Open-5" Handshapes (All fingers spread)',
    palmFacing: 'Both palms facing inward toward body',
    motionGuide: 'Both hands held forward at chest level, wiggling all fingers gently alternately.',
    facialNMM: 'Patient, calm facial expression.',
  },
  'I-LOVE-YOU': {
    handshape: 'ASL "I-L-Y" Handshape (Thumb, Index, Pinky extended; Middle and Ring folded)',
    palmFacing: 'Palm Forward / Outward toward recipient',
    motionGuide: 'Hold at chest/shoulder level with slight warm pulsation outward.',
    facialNMM: 'Warm affectionate smile and open eye contact.',
  },
  WATER: {
    handshape: 'ASL "W" Handshape (Index, Middle, Ring extended in W; Thumb holds Pinky)',
    palmFacing: 'Palm facing Left / Side',
    motionGuide: 'Index finger of W-hand taps chin twice gently.',
    facialNMM: 'Neutral attentive expression.',
  },
  ROOM: {
    handshape: 'Both hands "Open-B" (Flat walls)',
    palmFacing: 'Palms face each other for side walls, then front/back for end walls',
    motionGuide: 'Hands delineate the 4 perimeter walls of a room by boxing out parallel sides then rotating.',
    facialNMM: 'Focused spatial eye gaze following the perimeter.',
  },
  WHERE: {
    handshape: 'ASL "1" / Index Finger Extended',
    palmFacing: 'Palm Forward / Outward',
    motionGuide: 'Index finger wiggles side-to-side repeatedly at shoulder level.',
    facialNMM: 'Furrowed eyebrows, tilted head (Essential ASL WH-question grammatical rule).',
  },
  FOLLOW: {
    handshape: 'Both hands "10" / Thumbs-up A-fists',
    palmFacing: 'Palms face inward toward centerline',
    motionGuide: 'Dominant fist follows non-dominant fist forward in tandem, lead-and-follow trajectory.',
    facialNMM: 'Encouraging, directional head nod.',
  },
  WELCOME: {
    handshape: 'ASL "Open-B" Handshape (Curved welcoming palm)',
    palmFacing: 'Palm Upward and Inward',
    motionGuide: 'Hand sweeps smoothly from side inward toward the waist in a welcoming gesture.',
    facialNMM: 'Friendly welcoming smile.',
  },
  NAME: {
    handshape: 'Both hands "H" / Index + Middle fingers extended',
    palmFacing: 'Palms face inward towards centerline',
    motionGuide: 'Dominant H-fingers tap across non-dominant H-fingers twice at right angles.',
    facialNMM: 'Neutral or questioning depending on context.',
  },
  GOOD: {
    handshape: 'Dominant "Open-B" from lips to Non-dominant "Open-B" palm base',
    palmFacing: 'Palm inward at lips, falls flat into palm facing up',
    motionGuide: 'Fingertips touch lips, then drop smoothly into palm of flat non-dominant hand.',
    facialNMM: 'Pleasant smile and head nod.',
  },
};

// Helper: 3D point from MediaPipe landmark
function landmarkToVec3(pt: { x: number; y: number; z: number }, scale = 2.8): THREE.Vector3 {
  return new THREE.Vector3(
    (pt.x - 0.5) * scale,
    -(pt.y - 0.5) * scale,
    -(pt.z || 0) * scale * 1.2,
  );
}

// Helper: orient a cylinder between two 3D points
function orientCylinderBetweenPoints(
  mesh: THREE.Mesh,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  thickness: number,
) {
  const dir = new THREE.Vector3().subVectors(p2, p1);
  const length = dir.length();
  if (length < 0.0001) return;

  const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
  mesh.position.copy(mid);
  mesh.scale.set(thickness, length, thickness);

  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.clone().normalize(),
  );
  mesh.quaternion.copy(quat);
}

export default function SignPlayer({
  tokens,
  autoPlay = true,
  onSignComplete,
  height = 360,
  initialMode = '3d',
}: SignPlayerProps) {
  const [tokenIndex, setTokenIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playbackSpeed, setPlaybackSpeed] = useState<0.5 | 0.75 | 1.0 | 1.25>(1.0);
  const [isLooping, setIsLooping] = useState(true);
  const [viewMode, setViewMode] = useState<'3d' | 'skeleton'>(initialMode);

  // 3D Orbit Drag Rotation State
  const [orbitRot, setOrbitRot] = useState({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Three.js References
  const threeRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    dominantGroup: THREE.Group;
    secondaryGroup: THREE.Group;
    dominantJoints: THREE.Mesh[];
    dominantBones: THREE.Mesh[];
    secondaryJoints: THREE.Mesh[];
    secondaryBones: THREE.Mesh[];
    palmMeshDominant: THREE.Mesh;
    palmMeshSecondary: THREE.Mesh;
  } | null>(null);

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

  // Lookup ASL linguistic guidance for active sign
  const guideInfo = useMemo(() => {
    if (!activeToken) return null;
    if (ASL_HANDSHAPE_GUIDE[activeToken.gloss]) {
      return ASL_HANDSHAPE_GUIDE[activeToken.gloss];
    }
    if (activeToken.type === 'letter') {
      return {
        handshape: `ASL Fingerspelling Letter "${activeToken.gloss}"`,
        palmFacing: 'Palm Forward / Outward toward recipient (Standard Fingerspelling Rule)',
        motionGuide: `Hold hand steady at shoulder height, forming the canonical ASL "${activeToken.gloss}" handshape.`,
        facialNMM: 'Direct attentive eye contact with conversational partner.',
      };
    }
    if (activeToken.type === 'number') {
      return {
        handshape: `ASL Number Digit "${activeToken.gloss}"`,
        palmFacing: activeToken.gloss <= '5' ? 'Palm Inward (Traditional ASL 1-5)' : 'Palm Outward',
        motionGuide: `Form the precise "${activeToken.gloss}" digit configuration with clear finger extension.`,
        facialNMM: 'Neutral clear posture.',
      };
    }
    return {
      handshape: `ASL Vocabulary Sign "${activeToken.gloss}"`,
      palmFacing: 'Natural conversational orientation',
      motionGuide: activeToken.sign.description || 'Perform standard ASL gesture cleanly in the signing space.',
      facialNMM: 'Natural facial grammar in sync with gesture.',
    };
  }, [activeToken]);

  // Setup Three.js WebGL Scene
  useEffect(() => {
    if (Platform.OS !== 'web' || !canvasRef.current) return;
    const canvas = canvasRef.current;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(canvas.clientWidth || 360, canvas.clientHeight || height, false);
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.1;
    } catch {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, (canvas.clientWidth || 360) / (canvas.clientHeight || height), 0.1, 50);
    camera.position.set(0, 0, 3.4);
    camera.lookAt(0, 0, 0);

    // Realistic Multi-Point Studio Lighting (Apple / NVIDIA presentation standard)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    // Warm Key Light (Above front-right)
    const keyLight = new THREE.DirectionalLight(0xfff3e6, 1.8);
    keyLight.position.set(2.5, 4.0, 3.5);
    scene.add(keyLight);

    // Cyan Rim Fill Light (Gives crisp contour definition)
    const rimLight = new THREE.DirectionalLight(0x00e5ff, 0.7);
    rimLight.position.set(-3.0, -1.0, 2.0);
    scene.add(rimLight);

    // Subtle Lime Accent Light (GuideLens branding)
    const bounceLight = new THREE.DirectionalLight(0xb7ff56, 0.5);
    bounceLight.position.set(0.0, -3.5, 1.0);
    scene.add(bounceLight);

    // Materials: PBR Realistic Skin Shader
    const skinMaterial = new THREE.MeshStandardMaterial({
      color: 0xdeb494,
      roughness: 0.38,
      metalness: 0.05,
    });

    const jointMaterial = new THREE.MeshStandardMaterial({
      color: 0xd4a080,
      roughness: 0.32,
      metalness: 0.08,
    });

    const secondarySkinMaterial = new THREE.MeshStandardMaterial({
      color: 0xc8d8c0,
      roughness: 0.40,
      metalness: 0.05,
    });

    // Shared Geometries
    const sphereGeom = new THREE.SphereGeometry(1, 16, 16);
    const cylinderGeom = new THREE.CylinderGeometry(1, 1, 1, 16);

    // Build Dominant Hand Meshes
    const dominantGroup = new THREE.Group();
    scene.add(dominantGroup);

    const dominantJoints: THREE.Mesh[] = [];
    for (let i = 0; i < 21; i++) {
      const radius = i === 0 ? 0.12 : i === 4 || i === 8 || i === 12 || i === 16 || i === 20 ? 0.08 : 0.07;
      const mesh = new THREE.Mesh(sphereGeom, jointMaterial);
      mesh.scale.set(radius, radius, radius);
      dominantGroup.add(mesh);
      dominantJoints.push(mesh);
    }

    const dominantBones: THREE.Mesh[] = [];
    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
      const mesh = new THREE.Mesh(cylinderGeom, skinMaterial);
      dominantGroup.add(mesh);
      dominantBones.push(mesh);
    }

    // Palm hull mesh
    const palmGeom = new THREE.BufferGeometry();
    const palmMeshDominant = new THREE.Mesh(palmGeom, skinMaterial);
    dominantGroup.add(palmMeshDominant);

    // Build Secondary Hand Meshes
    const secondaryGroup = new THREE.Group();
    scene.add(secondaryGroup);

    const secondaryJoints: THREE.Mesh[] = [];
    for (let i = 0; i < 21; i++) {
      const radius = i === 0 ? 0.11 : 0.065;
      const mesh = new THREE.Mesh(sphereGeom, secondarySkinMaterial);
      mesh.scale.set(radius, radius, radius);
      secondaryGroup.add(mesh);
      secondaryJoints.push(mesh);
    }

    const secondaryBones: THREE.Mesh[] = [];
    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
      const mesh = new THREE.Mesh(cylinderGeom, secondarySkinMaterial);
      secondaryGroup.add(mesh);
      secondaryBones.push(mesh);
    }

    const palmMeshSecondary = new THREE.Mesh(new THREE.BufferGeometry(), secondarySkinMaterial);
    secondaryGroup.add(palmMeshSecondary);

    threeRef.current = {
      renderer,
      scene,
      camera,
      dominantGroup,
      secondaryGroup,
      dominantJoints,
      dominantBones,
      secondaryJoints,
      secondaryBones,
      palmMeshDominant,
      palmMeshSecondary,
    };

    return () => {
      dominantGroup.clear();
      secondaryGroup.clear();
      scene.clear();
      renderer.dispose();
      threeRef.current = null;
    };
  }, [height]);

  // Update 3D Hand Mesh positions from current landmarks
  const update3DHand = useCallback(
    (
      pose: HandPose,
      joints: THREE.Mesh[],
      bones: THREE.Mesh[],
      group: THREE.Group,
      isVisible: boolean,
    ) => {
      group.visible = isVisible;
      if (!isVisible || !pose || !pose.length) return;

      const vec3List: THREE.Vector3[] = [];
      for (let i = 0; i < 21; i++) {
        const pt = pose[i] || REST_POSE[i];
        const v = landmarkToVec3(pt);
        vec3List.push(v);
        joints[i].position.copy(v);
      }

      // Connect 3D bones with realistic phalanx tapering
      for (let k = 0; k < HAND_CONNECTIONS.length; k++) {
        const [start, end] = HAND_CONNECTIONS[k];
        const p1 = vec3List[start];
        const p2 = vec3List[end];
        if (!p1 || !p2) continue;

        // Thickness: metacarpal is thicker, proximal phalanx medium, distal tip tapered
        const thickness = end <= 4 ? 0.065 : end === 8 || end === 12 || end === 16 || end === 20 ? 0.05 : 0.062;
        orientCylinderBetweenPoints(bones[k], p1, p2, thickness);
      }
    },
    [],
  );

  // 2D Skeleton fallback renderer (MediaPipe X-Ray)
  const drawSkeletonHand = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      pose: HandPose,
      w: number,
      h: number,
      isSecondary = false,
    ) => {
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
      }
    },
    [],
  );

  // Main animation render loop
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
    // Paced slightly slower (~30%) for clear, deliberate human sign perception
    stateRef.current.keyframeDuration = ((keyframes[0].durationMs || 420) * 1.30) / playbackSpeed;

    const render = (now: number) => {
      const elapsed = now - stateRef.current.startTime;
      const duration = stateRef.current.keyframeDuration;
      const t = Math.min(1, Math.max(0, elapsed / duration));

      const activePose = interpolatePose(stateRef.current.currentPose, stateRef.current.targetPose, t);
      const secStart = stateRef.current.currentSecondary || stateRef.current.targetSecondary;
      const activeSecondary = stateRef.current.targetSecondary
        ? interpolatePose(secStart!, stateRef.current.targetSecondary, t)
        : undefined;

      if (viewMode === '3d' && threeRef.current) {
        const {
          renderer,
          scene,
          camera,
          dominantGroup,
          secondaryGroup,
          dominantJoints,
          dominantBones,
          secondaryJoints,
          secondaryBones,
        } = threeRef.current;

        // Apply interactive orbit rotation
        dominantGroup.rotation.set(orbitRot.x, orbitRot.y, 0);
        secondaryGroup.rotation.set(orbitRot.x, orbitRot.y, 0);

        // Update 3D meshes
        update3DHand(activePose, dominantJoints, dominantBones, dominantGroup, true);
        update3DHand(activeSecondary || REST_POSE, secondaryJoints, secondaryBones, secondaryGroup, Boolean(activeSecondary));

        renderer.render(scene, camera);
      } else if (canvasRef.current && viewMode === 'skeleton') {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
          const w = canvas.clientWidth || 360;
          const h = canvas.clientHeight || height;

          if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
            canvas.width = w * dpr;
            canvas.height = h * dpr;
          }

          ctx.save();
          ctx.scale(dpr, dpr);
          ctx.fillStyle = '#07120F';
          ctx.fillRect(0, 0, w, h);

          // Grid
          ctx.strokeStyle = 'rgba(183, 255, 86, 0.05)';
          ctx.lineWidth = 1;
          for (let gx = 0; gx < w; gx += 24) {
            ctx.beginPath();
            ctx.moveTo(gx, 0);
            ctx.lineTo(gx, h);
            ctx.stroke();
          }

          if (activeSecondary) {
            drawSkeletonHand(ctx, activeSecondary, w, h, true);
          }
          drawSkeletonHand(ctx, activePose, w, h, false);
          ctx.restore();
        }
      }

      // Check keyframe progress
      if (isPlaying && t >= 1) {
        const nextKeyframeIdx = stateRef.current.keyframeIdx + 1;
        if (nextKeyframeIdx < keyframes.length) {
          stateRef.current.keyframeIdx = nextKeyframeIdx;
          stateRef.current.currentPose = stateRef.current.targetPose;
          stateRef.current.targetPose = keyframes[nextKeyframeIdx].landmarks;
          stateRef.current.currentSecondary = stateRef.current.targetSecondary;
          stateRef.current.targetSecondary = keyframes[nextKeyframeIdx].secondaryLandmarks;
          // Paced slightly slower (~30%) for clear, deliberate human sign perception
          stateRef.current.keyframeDuration =
            ((keyframes[nextKeyframeIdx].durationMs || 420) * 1.30) / playbackSpeed;
          stateRef.current.startTime = now;
        } else {
          const nextTokenIdx = tokenIndex + 1;
          if (nextTokenIdx < tokens.length) {
            setTokenIndex(nextTokenIdx);
          } else {
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
    orbitRot,
    onSignComplete,
    update3DHand,
    drawSkeletonHand,
    height,
  ]);

  // Touch / Pointer Drag to rotate 3D Hand in Space
  const handlePointerDown = (e: any) => {
    isDraggingRef.current = true;
    const clientX = e.clientX ?? e.nativeEvent?.pageX ?? 0;
    const clientY = e.clientY ?? e.nativeEvent?.pageY ?? 0;
    dragStartRef.current = { x: clientX, y: clientY };
  };

  const handlePointerMove = (e: any) => {
    if (!isDraggingRef.current) return;
    const clientX = e.clientX ?? e.nativeEvent?.pageX ?? 0;
    const clientY = e.clientY ?? e.nativeEvent?.pageY ?? 0;
    const dx = clientX - dragStartRef.current.x;
    const dy = clientY - dragStartRef.current.y;
    dragStartRef.current = { x: clientX, y: clientY };

    setOrbitRot((prev) => ({
      x: Math.max(-0.9, Math.min(0.9, prev.x + dy * 0.008)),
      y: Math.max(-1.4, Math.min(1.4, prev.y + dx * 0.008)),
    }));
  };

  const handlePointerUp = () => {
    isDraggingRef.current = false;
  };

  const resetOrbit = () => {
    setOrbitRot({ x: 0, y: 0 });
  };

  const handlePrev = () => {
    setTokenIndex((curr) => Math.max(0, curr - 1));
  };

  const handleNext = () => {
    setTokenIndex((curr) => Math.min(tokens.length - 1, curr + 1));
  };

  const toggleSpeed = () => {
    setPlaybackSpeed((curr) => (curr === 1.0 ? 0.75 : curr === 0.75 ? 0.5 : curr === 0.5 ? 1.25 : 1.0));
  };

  if (!tokens.length) {
    return (
      <View style={[styles.emptyContainer, { height }]}>
        <Ionicons name="hand-left-outline" size={36} color="#4E6760" />
        <Text style={styles.emptyText}>Enter text or speak to generate sign translation</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 3D WebGL / Kinematic Landmark Viewport */}
      <View
        style={[styles.canvasWrapper, { height }]}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        {Platform.OS === 'web' ? (
          <canvas
            ref={canvasRef as any}
            style={{ width: '100%', height: '100%', display: 'block', cursor: 'grab' }}
          />
        ) : (
          <View style={styles.fallbackRig}>
            <Ionicons name="hand-left" size={64} color="#B7FF56" />
          </View>
        )}

        {/* Floating Gloss Banner & View Mode Toggle */}
        <View style={styles.glossOverlay} pointerEvents="box-none">
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

            {/* View Mode Switcher: True 3D WebGL PBR vs X-Ray */}
            <View style={styles.modePillGroup}>
              <Pressable
                onPress={() => setViewMode('3d')}
                style={[styles.viewModeToggle, viewMode === '3d' && styles.viewModeToggleActive]}
                accessibilityRole="button"
                accessibilityLabel="3D WebGL Studio view"
              >
                <Ionicons
                  name="cube"
                  size={12}
                  color={viewMode === '3d' ? '#07110F' : '#B7FF56'}
                />
                <Text
                  style={[
                    styles.viewModeToggleText,
                    viewMode === '3d' && styles.viewModeToggleTextActive,
                  ]}
                >
                  3D WebGL Hand
                </Text>
              </Pressable>

              <Pressable
                onPress={() => setViewMode('skeleton')}
                style={[styles.viewModeToggle, viewMode === 'skeleton' && styles.viewModeToggleActive]}
                accessibilityRole="button"
                accessibilityLabel="MediaPipe X-Ray Skeleton view"
              >
                <Ionicons
                  name="hardware-chip-outline"
                  size={12}
                  color={viewMode === 'skeleton' ? '#07110F' : '#B7FF56'}
                />
                <Text
                  style={[
                    styles.viewModeToggleText,
                    viewMode === 'skeleton' && styles.viewModeToggleTextActive,
                  ]}
                >
                  X-Ray
                </Text>
              </Pressable>
            </View>
          </View>

          <Text style={styles.glossTitle}>{activeToken.gloss}</Text>
          <Text style={styles.glossDesc} numberOfLines={2}>
            {activeToken.sign.description}
          </Text>

          {/* Interactive 3D Orbit Drag Hint */}
          {viewMode === '3d' && (
            <Pressable onPress={resetOrbit} style={styles.dragHintRow}>
              <Ionicons name="sync-outline" size={12} color="#8EA49B" />
              <Text style={styles.dragHintText}>
                Drag to rotate 3D hand angle in real time • Tap to reset
              </Text>
            </Pressable>
          )}
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

      {/* Authentic ASL Linguistic & Anatomical Signing Guide (Industry Standard) */}
      {guideInfo && (
        <View style={styles.guideCard}>
          <View style={styles.guideHeaderRow}>
            <View style={styles.guideTag}>
              <Ionicons name="book-outline" size={13} color="#B7FF56" />
              <Text style={styles.guideTagText}>ASL LINGUISTIC SIGNING GUIDE</Text>
            </View>
            <Text style={styles.guideGloss}>{activeToken.gloss}</Text>
          </View>

          {/* Handshape & Palm Orientation */}
          <View style={styles.guideSpecsGrid}>
            <View style={styles.guideSpecItem}>
              <View style={styles.specIconCircle}>
                <Ionicons name="hand-right" size={14} color="#00E5FF" />
              </View>
              <View style={styles.specCopy}>
                <Text style={styles.specLabel}>HANDSHAPE CONFIGURATION</Text>
                <Text style={styles.specValue}>{guideInfo.handshape}</Text>
              </View>
            </View>

            <View style={styles.guideSpecItem}>
              <View style={styles.specIconCircle}>
                <Ionicons name="compass-outline" size={14} color="#FFD043" />
              </View>
              <View style={styles.specCopy}>
                <Text style={styles.specLabel}>PALM ORIENTATION</Text>
                <Text style={styles.specValue}>{guideInfo.palmFacing}</Text>
              </View>
            </View>
          </View>

          {/* Physical Movement Trajectory */}
          <View style={styles.guideTrajectoryBox}>
            <View style={styles.trajectoryHeader}>
              <Ionicons name="arrow-forward-circle-outline" size={14} color="#C8FF72" />
              <Text style={styles.trajectoryTitle}>MOVEMENT TRAJECTORY & EXECUTION</Text>
            </View>
            <Text style={styles.trajectoryText}>{guideInfo.motionGuide}</Text>
          </View>

          {/* Non-Manual Markers (NMM / Face Expression) */}
          <View style={styles.guideNmmBox}>
            <View style={styles.trajectoryHeader}>
              <Ionicons name="happy-outline" size={14} color="#FF8A7A" />
              <Text style={[styles.trajectoryTitle, { color: '#FF8A7A' }]}>
                NON-MANUAL MARKERS (FACIAL & HEAD GRAMMAR)
              </Text>
            </View>
            <Text style={styles.nmmText}>{guideInfo.facialNMM}</Text>
          </View>
        </View>
      )}
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
    borderColor: '#19362A',
  },
  canvasWrapper: {
    width: '100%',
    position: 'relative',
    backgroundColor: '#050D0B',
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
  modePillGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(9, 21, 17, 0.85)',
    borderRadius: 8,
    padding: 2,
    borderWidth: 1,
    borderColor: 'rgba(183, 255, 86, 0.25)',
    gap: 3,
  },
  viewModeToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  viewModeToggleActive: {
    backgroundColor: '#B7FF56',
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
    textShadowColor: 'rgba(0, 0, 0, 0.9)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  glossDesc: {
    color: '#B8CCC5',
    fontSize: 12,
    marginTop: 2,
    lineHeight: 16,
    textShadowColor: 'rgba(0, 0, 0, 0.9)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  dragHintRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginTop: 4,
    backgroundColor: 'rgba(0, 0, 0, 0.45)',
    alignSelf: 'flex-start',
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 6,
  },
  dragHintText: {
    color: '#8EA49B',
    fontSize: 10,
    fontWeight: '500',
  },
  controlsBar: {
    position: 'absolute',
    bottom: 12,
    left: 14,
    right: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(7, 17, 15, 0.88)',
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
    backgroundColor: '#091512',
    borderTopWidth: 1,
    borderTopColor: '#162F24',
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
  guideCard: {
    backgroundColor: '#0B1914',
    padding: 14,
    gap: 10,
    borderTopWidth: 1,
    borderTopColor: '#173628',
  },
  guideHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  guideTag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  guideTagText: {
    color: '#B7FF56',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  guideGloss: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  guideSpecsGrid: {
    flexDirection: 'row',
    gap: 10,
  },
  guideSpecItem: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#07120E',
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#19362A',
    gap: 8,
  },
  specIconCircle: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  specCopy: {
    flex: 1,
  },
  specLabel: {
    color: '#769486',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: 2,
  },
  specValue: {
    color: '#E1EDE7',
    fontSize: 11,
    fontWeight: '700',
    lineHeight: 15,
  },
  guideTrajectoryBox: {
    backgroundColor: '#07120E',
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#19362A',
    gap: 4,
  },
  trajectoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  trajectoryTitle: {
    color: '#C8FF72',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  trajectoryText: {
    color: '#D2E2DB',
    fontSize: 12,
    lineHeight: 17,
  },
  guideNmmBox: {
    backgroundColor: '#07120E',
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#2D1E1E',
    gap: 4,
  },
  nmmText: {
    color: '#FFB8B8',
    fontSize: 12,
    lineHeight: 17,
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

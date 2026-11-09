import { getApiUrl } from './vision';
import type { LocalDetection } from './detector.types';

export type CampusAlert = {
  id: number;
  location: string;
  category: 'hazard' | 'obstacle' | 'sign' | 'room_status';
  label: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
  reported_by: string;
  created_at: string;
  relative_time: string;
  is_active: boolean;
};

export type WhatChangedResult = {
  has_changed: boolean;
  delta_summary: string;
  speech: string;
  added_objects: string[];
  removed_objects: string[];
  moved_objects: string[];
};

export type BridgeMessage = {
  id: number;
  session_id: string;
  sender_role: 'deaf' | 'blind';
  text: string;
  created_at: string;
  relative_time: string;
};

export const CAMPUS_LOCATIONS = [
  'Library Corridor',
  'Hallway 200',
  'Room 204',
  'Elevator B',
  'Main Campus Lobby',
  'Cafeteria',
] as const;

export type CampusLocation = (typeof CAMPUS_LOCATIONS)[number];

const SEED_ALERTS: CampusAlert[] = [
  {
    id: 1,
    location: 'Library Corridor',
    category: 'hazard',
    label: 'wet floor',
    description: 'Caution: Yellow wet-floor sign outside library entrance, reported recently.',
    severity: 'high',
    reported_by: 'student_a',
    created_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    relative_time: '5m ago',
    is_active: true,
  },
  {
    id: 2,
    location: 'Elevator B',
    category: 'room_status',
    label: 'out of order',
    description: 'Elevator B has an out-of-order service notice posted since 9 AM.',
    severity: 'medium',
    reported_by: 'student_campus',
    created_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    relative_time: '45m ago',
    is_active: true,
  },
  {
    id: 3,
    location: 'Hallway 200',
    category: 'obstacle',
    label: 'utility cart',
    description: 'A custodial cart is partially blocking the hallway walkway near Room 204.',
    severity: 'medium',
    reported_by: 'student_pat',
    created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    relative_time: '15m ago',
    is_active: true,
  },
  {
    id: 4,
    location: 'Room 204',
    category: 'room_status',
    label: 'closed',
    description: 'Room 204 reserved for hackathon judging and mentor reviews.',
    severity: 'low',
    reported_by: 'faculty',
    created_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    relative_time: '30m ago',
    is_active: true,
  },
];

const LOCAL_STORAGE_KEY_ALERTS = 'guidelens_campus_alerts';
const LOCAL_STORAGE_KEY_BASELINES = 'guidelens_room_baselines';
const LOCAL_STORAGE_KEY_BRIDGE = 'guidelens_bridge_messages';

const inMemoryCache: Record<string, string> = {};

function getStoredJson<T>(key: string, fallback: T): T {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const raw = window.localStorage.getItem(key);
      if (raw) return JSON.parse(raw) as T;
    }
  } catch {}
  if (inMemoryCache[key]) {
    try {
      return JSON.parse(inMemoryCache[key]) as T;
    } catch {}
  }
  return fallback;
}

function setStoredJson<T>(key: string, value: T): void {
  const serialized = JSON.stringify(value);
  inMemoryCache[key] = serialized;
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(key, serialized);
    }
  } catch {}
}

/**
 * Fetch active campus alerts.
 * Gracefully merges server state with local storage.
 * If server is unreachable or offline, returns local alerts with ZERO errors.
 */
export async function getCampusAlerts(location?: string): Promise<CampusAlert[]> {
  const localAlerts = getStoredJson<CampusAlert[]>(LOCAL_STORAGE_KEY_ALERTS, SEED_ALERTS);
  const apiUrl = getApiUrl();

  if (!apiUrl) {
    return filterAlerts(localAlerts, location);
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const query = location ? `?location=${encodeURIComponent(location)}` : '';
    const res = await fetch(`${apiUrl}/v1/memory/alerts${query}`, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    });
    clearTimeout(timeout);

    if (res.ok) {
      const serverAlerts = (await res.json()) as CampusAlert[];
      if (Array.isArray(serverAlerts) && serverAlerts.length > 0) {
        setStoredJson(LOCAL_STORAGE_KEY_ALERTS, serverAlerts);
        return filterAlerts(serverAlerts, location);
      }
    }
  } catch {
    // Offline or connection refused: silently fallback to local storage
  }

  return filterAlerts(localAlerts, location);
}

function filterAlerts(alerts: CampusAlert[], location?: string): CampusAlert[] {
  const active = alerts.filter((a) => a.is_active);
  if (!location || location.trim() === '' || location === 'All Campus') {
    return active;
  }
  const needle = location.toLowerCase();
  return active.filter(
    (a) => a.location.toLowerCase().includes(needle) || a.location === 'All Campus'
  );
}

/**
 * Report a new obstacle, hazard, or room status from this device.
 * Immediately saves locally, and syncs to DGX server if reachable.
 */
export async function reportCampusAlert(alert: {
  location: string;
  category: 'hazard' | 'obstacle' | 'sign' | 'room_status';
  label: string;
  description: string;
  severity?: 'low' | 'medium' | 'high';
}): Promise<CampusAlert> {
  const now = new Date();
  const newAlert: CampusAlert = {
    id: Date.now(),
    location: alert.location,
    category: alert.category,
    label: alert.label,
    description: alert.description,
    severity: alert.severity || 'medium',
    reported_by: 'this_device',
    created_at: now.toISOString(),
    relative_time: 'just now',
    is_active: true,
  };

  const localAlerts = getStoredJson<CampusAlert[]>(LOCAL_STORAGE_KEY_ALERTS, SEED_ALERTS);
  const updated = [newAlert, ...localAlerts];
  setStoredJson(LOCAL_STORAGE_KEY_ALERTS, updated);

  const apiUrl = getApiUrl();
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${apiUrl}/v1/memory/observe`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: alert.location,
          category: alert.category,
          label: alert.label,
          description: alert.description,
          severity: alert.severity || 'medium',
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        const serverItem = (await res.json()) as CampusAlert;
        return serverItem;
      }
    } catch {}
  }

  return newAlert;
}

/**
 * "What Changed?" — Compares currently detected objects against stored baseline memory.
 * Runs 100% locally on-device and can sync with server.
 */
export async function checkWhatChanged(
  location: string,
  currentDetections: LocalDetection[],
  updateBaseline = false
): Promise<WhatChangedResult> {
  const apiUrl = getApiUrl();

  // Try server first if configured
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${apiUrl}/v1/memory/what-changed`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location,
          current_objects: currentDetections.map((d) => ({
            label: d.label,
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
          })),
          update_baseline: updateBaseline,
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        return (await res.json()) as WhatChangedResult;
      }
    } catch {
      // Fallback to local on-device delta logic
    }
  }

  // 100% Offline on-device delta calculation
  const baselines = getStoredJson<Record<string, LocalDetection[]>>(
    LOCAL_STORAGE_KEY_BASELINES,
    {
      'Room 204': [
        { label: 'table', confidence: 0.9, x: 0.2, y: 0.5, width: 0.6, height: 0.4 },
        { label: 'chair', confidence: 0.85, x: 0.25, y: 0.55, width: 0.15, height: 0.25 },
      ],
      'Library Corridor': [
        { label: 'door', confidence: 0.88, x: 0.7, y: 0.2, width: 0.25, height: 0.7 },
      ],
    }
  );

  const baseline = baselines[location];
  if (!baseline || updateBaseline) {
    baselines[location] = currentDetections;
    setStoredJson(LOCAL_STORAGE_KEY_BASELINES, baselines);
    return {
      has_changed: false,
      delta_summary: `Baseline saved for ${location} with ${currentDetections.length} objects.`,
      speech: `Baseline recorded for ${location}. From now on, GuideLens will only announce what changes.`,
      added_objects: [],
      removed_objects: [],
      moved_objects: [],
    };
  }

  const baseLabels = baseline.map((b) => b.label.toLowerCase());
  const currLabels = currentDetections.map((c) => c.label.toLowerCase());

  const added: string[] = [];
  for (const label of currLabels) {
    if (baseLabels.indexOf(label) === -1 && added.indexOf(label) === -1) {
      added.push(label);
    }
  }

  const removed: string[] = [];
  for (const label of baseLabels) {
    if (currLabels.indexOf(label) === -1 && removed.indexOf(label) === -1) {
      removed.push(label);
    }
  }

  const moved: string[] = [];
  for (const curr of currentDetections) {
    const match = baseline.find((b) => b.label.toLowerCase() === curr.label.toLowerCase());
    if (match && Math.abs(curr.x - match.x) > 0.22) {
      if (moved.indexOf(curr.label) === -1) moved.push(curr.label);
    }
  }

  const hasChanged = added.length > 0 || moved.length > 0;
  if (!hasChanged) {
    return {
      has_changed: false,
      delta_summary: 'No changes detected since earlier.',
      speech: `All objects in ${location} remain in their usual positions. Your path is clear.`,
      added_objects: [],
      removed_objects: [],
      moved_objects: [],
    };
  }

  const parts: string[] = [];
  if (added.length > 0) {
    parts.push(`Something new is in your path: ${added.slice(0, 2).join(', ')}`);
  }
  if (moved.length > 0) {
    parts.push(`A ${moved.slice(0, 2).join(', ')} on your usual path has moved`);
  }

  return {
    has_changed: true,
    delta_summary: parts.join('; '),
    speech: `Notice in ${location}: ${parts.join('. ')}.`,
    added_objects: added,
    removed_objects: removed,
    moved_objects: moved,
  };
}

/**
 * Deaf-Blind Bridge message management.
 */
export async function sendBridgeMessage(
  sessionId: string,
  senderRole: 'deaf' | 'blind',
  text: string
): Promise<BridgeMessage> {
  const newMsg: BridgeMessage = {
    id: Date.now(),
    session_id: sessionId,
    sender_role: senderRole,
    text: text.trim(),
    created_at: new Date().toISOString(),
    relative_time: 'just now',
  };

  const stored = getStoredJson<BridgeMessage[]>(LOCAL_STORAGE_KEY_BRIDGE, []);
  setStoredJson(LOCAL_STORAGE_KEY_BRIDGE, [...stored, newMsg].slice(-50));

  const apiUrl = getApiUrl();
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      const res = await fetch(`${apiUrl}/v1/bridge/message`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          sender_role: senderRole,
          text: text.trim(),
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        return (await res.json()) as BridgeMessage;
      }
    } catch {}
  }

  return newMsg;
}

export async function getBridgeMessages(sessionId: string): Promise<BridgeMessage[]> {
  const local = getStoredJson<BridgeMessage[]>(LOCAL_STORAGE_KEY_BRIDGE, []);
  const apiUrl = getApiUrl();
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      const res = await fetch(`${apiUrl}/v1/bridge/messages?session_id=${encodeURIComponent(sessionId)}`, {
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
      });
      clearTimeout(timeout);
      if (res.ok) {
        const msgs = (await res.json()) as BridgeMessage[];
        if (Array.isArray(msgs) && msgs.length > 0) {
          setStoredJson(LOCAL_STORAGE_KEY_BRIDGE, msgs);
          return msgs;
        }
      }
    } catch {}
  }
  return local.filter((m) => m.session_id === sessionId);
}

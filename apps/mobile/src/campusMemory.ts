import { getApiUrl } from './vision';
import { getSightDeviceId, getSightHeaders, getSightSessionId } from './sightIdentity';
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

export type BridgeRole = 'deaf' | 'blind';

export type BridgeMessage = {
  /** Server id; negative for a message that has not reached the server yet. */
  id: number;
  /** The conversation code. */
  session_id: string;
  sender_role: BridgeRole;
  text: string;
  client_id?: string | null;
  from_this_device?: boolean;
  created_at: string;
  relative_time: string;
  /** True while the message waits in this device's outbox. */
  pending?: boolean;
};

export type BridgeConversation = {
  code: string;
  created_at: string;
  expires_at: string;
  devices: number;
  max_devices: number;
  partner_joined: boolean;
  is_member: boolean;
};

export type SightObservation = {
  id: number;
  category: string;
  label: string;
  description: string;
  zone_id: string;
  timestamp: string;
  source_device: string;
  source_session: string;
  confidence: number;
  verification_status: 'verified' | 'unverified';
  privacy_scope: 'personal' | 'shared';
  room_x: number | null;
  room_y: number | null;
  expires_at: string;
  origin: 'manual' | 'detector';
};

export type SightDashboard = {
  zone_id: string;
  connected_devices: number;
  recent_observations: SightObservation[];
  tracking_latency_ms: number | null;
  tracking_requests_last_5m: number;
};

export type SightChanges = {
  zone_id: string;
  generated_at: string;
  compared_seconds: number;
  added: string[];
  previously_seen: string[];
  summary: string;
  verified_only: boolean;
};

export const currentSightSession = getSightSessionId;
export const currentSightDevice = getSightDeviceId;

async function sightJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const apiUrl = getApiUrl();
  if (!apiUrl) throw new Error('Connect the DGX Spark API to use Shared Sight');
  const headers = new Headers(getSightHeaders());
  new Headers(init.headers).forEach((value, key) => headers.set(key, value));
  const response = await fetch(`${apiUrl}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail || `Shared Sight request failed (${response.status})`);
  }
  return await response.json() as T;
}

export async function submitSightObservation(input: {
  zone_id: string;
  category: string;
  label: string;
  description?: string;
  confidence: number;
  privacy_scope: 'personal' | 'shared';
  verified: boolean;
  room_x?: number;
  room_y?: number;
  origin?: 'manual' | 'detector';
}): Promise<SightObservation> {
  return sightJson<SightObservation>('/v1/observations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
}

export async function getSightDashboard(zoneId: string): Promise<SightDashboard> {
  return sightJson<SightDashboard>(`/v1/shared-sight/dashboard?zone_id=${encodeURIComponent(zoneId)}`);
}

export async function getSightChanges(zoneId: string): Promise<SightChanges> {
  return sightJson<SightChanges>(`/v1/observations/changes?zone_id=${encodeURIComponent(zoneId)}`);
}

export async function searchPersonalRecall(query: string): Promise<SightObservation[]> {
  return sightJson<SightObservation[]>(`/v1/personal/observations?query=${encodeURIComponent(query)}`);
}

export async function removeSightObservation(id: number): Promise<void> {
  await sightJson<{ status: string }>(`/v1/observations/${id}`, { method: 'DELETE' });
}

export const CAMPUS_LOCATIONS = [
  'Library Corridor',
  'Hallway 200',
  'Room 204',
  'Elevator B',
  'Main Campus Lobby',
  'Cafeteria',
] as const;

export type CampusLocation = (typeof CAMPUS_LOCATIONS)[number];

const SEED_ALERTS: CampusAlert[] = [];

const LOCAL_STORAGE_KEY_ALERTS = 'guidelens_campus_alerts_v2';
const LOCAL_STORAGE_KEY_BASELINES = 'guidelens_room_baselines';

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
      headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
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
        headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
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
        headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
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
 * Deaf-Blind Bridge.
 *
 * A conversation is one pair of devices joined by a 6-digit code, or a single
 * device with no sync (LOCAL_BRIDGE_CODE). Every message is written to this
 * device first and sent from a small outbox, so a dropped connection never
 * loses a message and a retry is stored once (the server dedupes client_id).
 */
export const LOCAL_BRIDGE_CODE = 'local';
const BRIDGE_STORAGE_PREFIX = 'guidelens_bridge_v2:';
const BRIDGE_REQUEST_TIMEOUT_MS = 3000;
const BRIDGE_LOCAL_HISTORY = 100;

export class BridgeGoneError extends Error {}

type StoredBridge = { messages: BridgeMessage[] };

function bridgeKey(code: string) {
  return `${BRIDGE_STORAGE_PREFIX}${code}`;
}

function readBridge(code: string): StoredBridge {
  const stored = getStoredJson<StoredBridge>(bridgeKey(code), { messages: [] });
  return Array.isArray(stored?.messages) ? stored : { messages: [] };
}

function writeBridge(code: string, value: StoredBridge) {
  setStoredJson(bridgeKey(code), { messages: value.messages.slice(-BRIDGE_LOCAL_HISTORY) });
}

async function bridgeRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const apiUrl = getApiUrl();
  if (!apiUrl) throw new Error('No GuideLens server is configured');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), BRIDGE_REQUEST_TIMEOUT_MS);
  try {
    const headers = new Headers(getSightHeaders());
    headers.set('Content-Type', 'application/json');
    const response = await fetch(`${apiUrl}${path}`, { ...init, headers, signal: controller.signal });
    if (response.status === 403 || response.status === 404 || response.status === 409) {
      const body = await response.json().catch(() => null) as { detail?: string } | null;
      throw new BridgeGoneError(body?.detail || 'That bridge conversation is not available');
    }
    if (!response.ok) throw new Error(`Bridge request failed (${response.status})`);
    return await response.json() as T;
  } finally {
    clearTimeout(timeout);
  }
}

export function bridgeServerConfigured(): boolean {
  return Boolean(getApiUrl());
}

export function createBridgeConversation(): Promise<BridgeConversation> {
  return bridgeRequest<BridgeConversation>('/v1/bridge/conversations', { method: 'POST' });
}

export function joinBridgeConversation(code: string): Promise<BridgeConversation> {
  return bridgeRequest<BridgeConversation>(`/v1/bridge/conversations/${encodeURIComponent(code)}/join`, { method: 'POST' });
}

export function getBridgeConversation(code: string): Promise<BridgeConversation> {
  return bridgeRequest<BridgeConversation>(`/v1/bridge/conversations/${encodeURIComponent(code)}`);
}

function sameMessage(left: BridgeMessage, right: BridgeMessage) {
  if (left.client_id && right.client_id) return left.client_id === right.client_id;
  return left.id === right.id;
}

/** Merge by client_id (or id); server messages in id order, unsent ones last. */
export function mergeBridgeMessages(existing: BridgeMessage[], incoming: BridgeMessage[]): BridgeMessage[] {
  const merged = [...existing];
  incoming.forEach((message) => {
    const index = merged.findIndex((item) => sameMessage(item, message));
    if (index >= 0) merged[index] = { ...merged[index], ...message, pending: message.pending ?? false };
    else merged.push(message);
  });
  const sent = merged.filter((item) => !item.pending).sort((left, right) => left.id - right.id);
  const pending = merged.filter((item) => item.pending);
  return [...sent, ...pending];
}

/** This device's copy of a conversation, including messages not yet sent. */
export function loadLocalBridgeMessages(code: string): BridgeMessage[] {
  return readBridge(code).messages;
}

function storeBridgeMessages(code: string, incoming: BridgeMessage[]) {
  const stored = readBridge(code);
  writeBridge(code, { messages: mergeBridgeMessages(stored.messages, incoming) });
}

function newClientId() {
  return `${getSightDeviceId().slice(0, 24)}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

async function deliver(code: string, message: BridgeMessage): Promise<BridgeMessage> {
  const sent = await bridgeRequest<BridgeMessage>('/v1/bridge/message', {
    method: 'POST',
    body: JSON.stringify({
      session_id: code,
      sender_role: message.sender_role,
      text: message.text,
      client_id: message.client_id,
    }),
  });
  const delivered = { ...sent, pending: false };
  storeBridgeMessages(code, [delivered]);
  return delivered;
}

/**
 * Save a message on this device and try to send it once. When the send fails
 * the message stays pending; flushBridgeOutbox retries it later.
 */
export async function sendBridgeMessage(code: string, senderRole: BridgeRole, text: string): Promise<BridgeMessage> {
  const now = new Date();
  const local: BridgeMessage = {
    id: -now.getTime(),
    session_id: code,
    sender_role: senderRole,
    text: text.trim(),
    client_id: newClientId(),
    from_this_device: true,
    created_at: now.toISOString(),
    relative_time: 'just now',
    pending: code !== LOCAL_BRIDGE_CODE,
  };
  storeBridgeMessages(code, [local]);
  if (code === LOCAL_BRIDGE_CODE || !bridgeServerConfigured()) return local;
  try {
    return await deliver(code, local);
  } catch (error) {
    if (error instanceof BridgeGoneError) throw error;
    return local;
  }
}

/** Retry unsent messages in order. Returns the ones that went through. */
export async function flushBridgeOutbox(code: string): Promise<BridgeMessage[]> {
  if (code === LOCAL_BRIDGE_CODE || !bridgeServerConfigured()) return [];
  const delivered: BridgeMessage[] = [];
  for (const message of readBridge(code).messages.filter((item) => item.pending)) {
    delivered.push(await deliver(code, message));
  }
  return delivered;
}

/** Newest messages from the server, or only those after afterId. Throws when offline. */
export async function fetchBridgeMessages(code: string, afterId?: number): Promise<BridgeMessage[]> {
  const query = afterId != null && afterId > 0 ? `&after_id=${afterId}` : '';
  const messages = await bridgeRequest<BridgeMessage[]>(
    `/v1/bridge/messages?session_id=${encodeURIComponent(code)}${query}&limit=50`,
  );
  const received = messages.map((message) => ({ ...message, pending: false }));
  storeBridgeMessages(code, received);
  return received;
}

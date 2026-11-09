let memoryDeviceId = '';
let memorySessionId = '';

function makeId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function readOrCreate(key: string, prefix: string) {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const existing = window.localStorage.getItem(key);
      if (existing) return existing;
      const next = makeId(prefix);
      window.localStorage.setItem(key, next);
      return next;
    }
  } catch {}
  return makeId(prefix);
}

export function getSightDeviceId() {
  if (!memoryDeviceId) memoryDeviceId = readOrCreate('guidelens_sight_device_id', 'device');
  return memoryDeviceId;
}

export function getSightSessionId() {
  if (!memorySessionId) memorySessionId = makeId('session');
  return memorySessionId;
}

export function getSightHeaders(): Record<string, string> {
  let key = process.env.EXPO_PUBLIC_SHARED_SIGHT_TOKEN || '';
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      key = window.localStorage.getItem('guidelens_shared_sight_key') || key;
    }
  } catch {}
  return {
    'X-Device-ID': getSightDeviceId(),
    'X-Session-ID': getSightSessionId(),
    ...(key ? { 'X-Shared-Sight-Key': key } : {}),
  };
}

import * as Crypto from 'expo-crypto';

let memoryDeviceId = '';
let memorySessionId = '';
let memoryDeviceSecret = '';

function makeId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function makeDeviceSecret() {
  const bytes = Crypto.getRandomBytes(32);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
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
  if (!memoryDeviceId) {
    memoryDeviceId = `device-${getSightDeviceSecret().slice(0, 32)}`;
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem('guidelens_sight_device_id', memoryDeviceId);
      }
    } catch {}
  }
  return memoryDeviceId;
}

export function getSightSessionId() {
  if (!memorySessionId) memorySessionId = makeId('session');
  return memorySessionId;
}

export function getSightDeviceSecret() {
  if (!memoryDeviceSecret) {
    memoryDeviceSecret = readOrCreate('guidelens_sight_device_secret', 'unused');
    if (!/^[a-f0-9]{64}$/.test(memoryDeviceSecret)) {
      memoryDeviceSecret = makeDeviceSecret();
      try {
        if (typeof window !== 'undefined' && window.localStorage) {
          window.localStorage.setItem('guidelens_sight_device_secret', memoryDeviceSecret);
        }
      } catch {}
    }
  }
  return memoryDeviceSecret;
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
    'X-Device-Secret': getSightDeviceSecret(),
    ...(key ? { 'X-Shared-Sight-Key': key } : {}),
  };
}

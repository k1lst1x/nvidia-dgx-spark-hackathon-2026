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

export function sha256Sync(ascii: string): string {
  function rightRotate(value: number, amount: number) {
    return (value >>> amount) | (value << (32 - amount));
  }
  const words: number[] = [];
  const asciiBitLength = ascii.length * 8;
  const hash = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  const k = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  for (let i = 0; i < ascii.length; i++) {
    words[i >> 2] |= (ascii.charCodeAt(i) & 0xff) << (8 * (3 - (i % 4)));
  }
  words[asciiBitLength >> 5] |= 0x80 << (24 - (asciiBitLength % 32));
  words[(((asciiBitLength + 64) >> 9) << 4) + 15] = asciiBitLength;
  const w = new Array(64);
  for (let i = 0; i < words.length; i += 16) {
    let [ha, hb, hc, hd, he, hf, hg, hh] = hash;
    for (let j = 0; j < 64; j++) {
      if (j < 16) {
        w[j] = words[i + j] | 0;
      } else {
        const s0 = rightRotate(w[j - 15], 7) ^ rightRotate(w[j - 15], 18) ^ (w[j - 15] >>> 3);
        const s1 = rightRotate(w[j - 2], 17) ^ rightRotate(w[j - 2], 19) ^ (w[j - 2] >>> 10);
        w[j] = (w[j - 16] + s0 + w[j - 7] + s1) | 0;
      }
      const s1 = rightRotate(he, 6) ^ rightRotate(he, 11) ^ rightRotate(he, 25);
      const ch = (he & hf) ^ (~he & hg);
      const temp1 = (hh + s1 + ch + k[j] + w[j]) | 0;
      const s0 = rightRotate(ha, 2) ^ rightRotate(ha, 13) ^ rightRotate(ha, 22);
      const maj = (ha & hb) ^ (ha & hc) ^ (hb & hc);
      const temp2 = (s0 + maj) | 0;
      hh = hg;
      hg = hf;
      hf = he;
      he = (hd + temp1) | 0;
      hd = hc;
      hc = hb;
      hb = ha;
      ha = (temp1 + temp2) | 0;
    }
    hash[0] = (hash[0] + ha) | 0;
    hash[1] = (hash[1] + hb) | 0;
    hash[2] = (hash[2] + hc) | 0;
    hash[3] = (hash[3] + hd) | 0;
    hash[4] = (hash[4] + he) | 0;
    hash[5] = (hash[5] + hf) | 0;
    hash[6] = (hash[6] + hg) | 0;
    hash[7] = (hash[7] + hh) | 0;
  }
  let result = '';
  for (let i = 0; i < 8; i++) {
    for (let j = 3; j >= 0; j--) {
      const byte = (hash[i] >> (j * 8)) & 0xff;
      result += byte.toString(16).padStart(2, '0');
    }
  }
  return result;
}

export function getSightDeviceId() {
  if (!memoryDeviceId) {
    const secret = getSightDeviceSecret();
    memoryDeviceId = `device-${sha256Sync(secret).slice(0, 32)}`;
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

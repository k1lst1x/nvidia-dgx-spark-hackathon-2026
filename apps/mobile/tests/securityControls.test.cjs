const test = require('node:test');
const assert = require('node:assert');
const crypto = require('node:crypto');
const loadTs = require('./load-ts.cjs');

const mockCrypto = {
  getRandomBytes: (size) => crypto.randomBytes(size),
};

const identity = loadTs('src/sightIdentity.ts', {
  'expo-crypto': mockCrypto,
});

const vision = loadTs('src/vision.ts', {
  'expo-crypto': mockCrypto,
  './detector': {
    detectObjects: () => Promise.resolve([]),
    detectVideoFrame: () => Promise.resolve(null),
    localDetectorBackend: () => 'stub',
    warmUpDetector: () => Promise.resolve(),
  },
  './detector.types': {},
  './yoloxEngine': {},
  './cameraCoach': {},
});

test('sha256Sync matches node crypto sha256', () => {
  const secret = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
  const expected = crypto.createHash('sha256').update(secret).digest('hex');
  const actual = identity.sha256Sync(secret);
  assert.strictEqual(actual, expected);
});

test('getSightDeviceId derives from sha256 of full secret without leaking secret prefix', () => {
  const secret = identity.getSightDeviceSecret();
  assert.match(secret, /^[a-f0-9]{64}$/);

  const deviceId = identity.getSightDeviceId();
  const expectedHash = crypto.createHash('sha256').update(secret).digest('hex').slice(0, 32);
  assert.strictEqual(deviceId, `device-${expectedHash}`);

  // Crucial security property: device ID must NEVER equal the raw secret prefix
  assert.notStrictEqual(deviceId, `device-${secret.slice(0, 32)}`);
});

test('tampering with secret suffix alters the derived device ID completely', () => {
  const originalSecret = 'a'.repeat(64);
  const tamperedSecret = 'a'.repeat(32) + 'b'.repeat(32);

  const originalId = `device-${identity.sha256Sync(originalSecret).slice(0, 32)}`;
  const tamperedId = `device-${identity.sha256Sync(tamperedSecret).slice(0, 32)}`;

  assert.notStrictEqual(originalId, tamperedId);
});

test('isSecureEndpoint enforces transport security', () => {
  // Loopback HTTP allowed for local developer debugging
  assert.strictEqual(vision.isSecureEndpoint('http://localhost:8000'), true);
  assert.strictEqual(vision.isSecureEndpoint('http://127.0.0.1:8000'), true);

  // HTTPS allowed for all endpoints
  assert.strictEqual(vision.isSecureEndpoint('https://api.guidelens.org'), true);
  assert.strictEqual(vision.isSecureEndpoint('https://dgx-spark.local:8000'), true);

  // Cleartext HTTP rejected for remote / LAN endpoints
  assert.strictEqual(vision.isSecureEndpoint('http://dgx-spark.local:8000'), false);
  assert.strictEqual(vision.isSecureEndpoint('http://192.168.1.50:8000'), false);
  assert.strictEqual(vision.isSecureEndpoint('http://campus-vision.internal'), false);
  assert.strictEqual(vision.isSecureEndpoint('ftp://localhost'), false);
});

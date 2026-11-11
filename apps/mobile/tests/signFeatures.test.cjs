const test = require('node:test');
const assert = require('node:assert');
const f = require('./load-ts.cjs')('src/signFeatures.ts');

const points = (count, base) => Array.from({ length: count }, (_, i) => ({ x: base + i, y: base + i + 0.25, z: base + i + 0.5 }));

test('the blendshape order is the 52 MediaPipe names, alphabetical, in any locale', () => {
  assert.strictEqual(f.BLENDSHAPE_ORDER.length, 52);
  assert.strictEqual(new Set(f.BLENDSHAPE_ORDER).size, 52);
  // Same order the tracker produced before the list was pinned (localeCompare)...
  assert.deepStrictEqual([...f.BLENDSHAPE_ORDER].sort((a, b) => a.localeCompare(b)), [...f.BLENDSHAPE_ORDER]);
  // ...and the same as a plain code-point sort, which a Python producer would use.
  assert.deepStrictEqual([...f.BLENDSHAPE_ORDER].sort(), [...f.BLENDSHAPE_ORDER]);
});

test('test vector: every block lands at its documented index', () => {
  const frame = f.buildSignFeatureFrame({
    leftHand: points(21, 100), rightHand: points(21, 200), pose: points(33, 300), facePresent: true,
    // Deliberately out of order, with one unknown name.
    blendshapes: [{ categoryName: 'noseSneerRight', score: 0.9 }, { categoryName: 'jawOpen', score: 0.5 },
      { categoryName: '_neutral', score: 0.1 }, { categoryName: 'notABlendshape', score: 1 }],
  });
  assert.strictEqual(frame.length, 281);
  assert.deepStrictEqual(frame.slice(0, 3), [100, 100.25, 100.5]);       // left hand, landmark 0
  assert.deepStrictEqual(frame.slice(60, 63), [120, 120.25, 120.5]);     // left hand, landmark 20
  assert.deepStrictEqual(frame.slice(63, 66), [200, 200.25, 200.5]);     // right hand, landmark 0
  assert.deepStrictEqual(frame.slice(126, 129), [300, 300.25, 300.5]);   // pose, landmark 0
  assert.deepStrictEqual(frame.slice(222, 225), [332, 332.25, 332.5]);   // pose, landmark 32
  assert.strictEqual(frame[225], 0.1);                                   // _neutral
  assert.strictEqual(frame[225 + 25], 0.5);                              // jawOpen
  assert.strictEqual(frame[276], 0.9);                                   // noseSneerRight
  assert.strictEqual(frame.slice(225, 277).filter((v) => v !== 0).length, 3);
  assert.deepStrictEqual(frame.slice(277), [1, 1, 1, 1]);
});

test('missing parts are zeros with a zero presence flag', () => {
  const frame = f.buildSignFeatureFrame({ leftHand: null, rightHand: points(21, 1), pose: null, facePresent: false });
  assert.ok(frame.slice(0, 63).every((v) => v === 0));
  assert.ok(frame.slice(126, 277).every((v) => v === 0));
  assert.deepStrictEqual(frame.slice(277), [0, 1, 0, 0]);
  assert.ok(frame.every(Number.isFinite));
});

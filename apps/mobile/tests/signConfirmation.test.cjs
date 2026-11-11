const test = require('node:test');
const assert = require('node:assert');
const c = require('./load-ts.cjs')('src/signConfirmation.ts');

const guess = (label, confidence) => ({ label, confidence });

test('a clear top guess gets a vote', () => {
  assert.strictEqual(c.windowVote([guess('water', 0.62), guess('milk', 0.2)]), 'water');
  assert.strictEqual(c.windowVote([guess('water', 0.31)]), 'water');
});

test('a guess below the floor gets no vote', () => {
  assert.strictEqual(c.windowVote([guess('water', 0.29), guess('milk', 0.05)]), '');
  assert.strictEqual(c.windowVote([]), '');
});

test('a near-tie with the runner-up gets no vote, however high', () => {
  assert.strictEqual(c.windowVote([guess('water', 0.48), guess('milk', 0.41)]), '');
  assert.strictEqual(c.windowVote([guess('water', 0.36), guess('milk', 0.3)]), '');
});

test('confident guesses need two agreeing windows, others three', () => {
  assert.strictEqual(c.requiredStableWindows(0.45), 2);
  assert.strictEqual(c.requiredStableWindows(0.44), 3);
});

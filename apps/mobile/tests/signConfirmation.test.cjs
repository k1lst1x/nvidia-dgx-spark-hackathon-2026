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

test('a word outside the allowed vocabulary gets no vote', () => {
  assert.strictEqual(c.windowVote([guess('duck', 0.95), guess('water', 0.01)], c.CORE_SIGNS), '');
  assert.strictEqual(c.windowVote([guess('water', 0.95), guess('duck', 0.01)], c.CORE_SIGNS), 'water');
  assert.strictEqual(c.windowVote([guess('duck', 0.95), guess('water', 0.01)]), 'duck');
});

test('every core sign is a label of the word model', () => {
  const { classes } = require('../assets/models/asl_word_labels.json');
  assert.strictEqual(c.CORE_SIGNS.size, 45);
  assert.deepStrictEqual([...c.CORE_SIGNS].filter((sign) => !classes.includes(sign)), []);
});

test('the core vocabulary is the default', () => {
  assert.strictEqual(c.readSignVocabulary(), 'core');
});

test('confident guesses need two agreeing windows, others three', () => {
  assert.strictEqual(c.requiredStableWindows(0.45), 2);
  assert.strictEqual(c.requiredStableWindows(0.44), 3);
});

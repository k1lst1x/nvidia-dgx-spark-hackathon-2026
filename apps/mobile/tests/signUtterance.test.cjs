const test = require('node:test');
const assert = require('node:assert');
const { UtteranceController } = require('./load-ts.cjs')('src/signUtterance.ts');

/** Drive the controller at 10 Hz. */
function run(controller, from, to, input) {
  let state;
  for (let now = from; now <= to; now += 100) state = controller.update({ now, commitCount: 0, ...input });
  return state;
}

test('stays idle until hands appear, then signs', () => {
  const c = new UtteranceController();
  assert.strictEqual(run(c, 0, 1000, { handsVisible: false }).phase, 'idle');
  assert.strictEqual(run(c, 1100, 1500, { handsVisible: true }).phase, 'signing');
});

test('words are provisional while signing and final only after the grace period', () => {
  const c = new UtteranceController();
  run(c, 0, 1000, { handsVisible: true });
  let s = c.update({ now: 1100, handsVisible: true, commitCount: 1, latestWord: 'thankyou' });
  s = c.update({ now: 2000, handsVisible: true, commitCount: 2, latestWord: 'help' });
  assert.deepStrictEqual(s.words, ['thankyou', 'help']);
  assert.strictEqual(s.phase, 'signing');
  assert.strictEqual(s.finalizedCount, 0);
  s = run(c, 2100, 3200, { handsVisible: false, commitCount: 2 });
  assert.strictEqual(s.phase, 'possible_end');
  assert.ok(s.endProgress > 0 && s.endProgress < 1);
  s = run(c, 3300, 4700, { handsVisible: false, commitCount: 2 });
  assert.strictEqual(s.phase, 'finalized');
  assert.strictEqual(s.outcome, 'recognized');
  assert.deepStrictEqual(s.words, ['thankyou', 'help']);
  assert.strictEqual(s.finalizedCount, 1);
  // Staying idle afterwards does not finalize again (no repeated speech).
  assert.strictEqual(run(c, 4800, 9000, { handsVisible: false, commitCount: 2 }).finalizedCount, 1);
});

test('a pause inside a sentence does not end it', () => {
  const c = new UtteranceController();
  run(c, 0, 1000, { handsVisible: true });
  c.update({ now: 1100, handsVisible: true, commitCount: 1, latestWord: 'where' });
  assert.strictEqual(run(c, 1200, 2600, { handsVisible: false, commitCount: 1 }).phase, 'possible_end');
  let s = run(c, 2700, 3000, { handsVisible: true, commitCount: 1 });
  assert.strictEqual(s.phase, 'signing');
  s = c.update({ now: 3100, handsVisible: true, commitCount: 2, latestWord: 'water' });
  assert.deepStrictEqual(s.words, ['where', 'water']);
  assert.strictEqual(s.finalizedCount, 0);
});

test('Finish sentence finalizes at once, and only once', () => {
  const c = new UtteranceController();
  run(c, 0, 1000, { handsVisible: true });
  c.update({ now: 1100, handsVisible: true, commitCount: 1, latestWord: 'yes' });
  const s = c.finish();
  assert.strictEqual(s.phase, 'finalized');
  assert.deepStrictEqual(s.words, ['yes']);
  assert.strictEqual(c.finish().finalizedCount, 1);
});

test('signing with nothing confirmed ends as unrecognized, not as invented text', () => {
  const c = new UtteranceController();
  run(c, 0, 2500, { handsVisible: true });
  const s = run(c, 2600, 5500, { handsVisible: false });
  assert.strictEqual(s.phase, 'finalized');
  assert.strictEqual(s.outcome, 'unrecognized');
  assert.deepStrictEqual(s.words, []);
});

test('a hand passing through the frame is ignored', () => {
  const c = new UtteranceController();
  run(c, 0, 300, { handsVisible: true });
  const s = run(c, 400, 4000, { handsVisible: false });
  assert.strictEqual(s.phase, 'idle');
  assert.strictEqual(s.finalizedCount, 0);
});

test('the next utterance starts clean and a cleared recognizer adds no word', () => {
  const c = new UtteranceController();
  run(c, 0, 1000, { handsVisible: true });
  c.update({ now: 1100, handsVisible: true, commitCount: 3, latestWord: 'hello' });
  c.finish();
  // Recognizer cleared: its counter drops to 0. No word may appear from that.
  let s = c.update({ now: 2000, handsVisible: true, commitCount: 0, latestWord: 'hello' });
  assert.deepStrictEqual(s.words, []);
  s = c.update({ now: 2100, handsVisible: true, commitCount: 1, latestWord: 'no' });
  assert.deepStrictEqual(s.words, ['no']);
});

test('the same sign confirmed twice is kept twice', () => {
  const c = new UtteranceController();
  run(c, 0, 500, { handsVisible: true });
  c.update({ now: 600, handsVisible: true, commitCount: 1, latestWord: 'more' });
  const s = c.update({ now: 4000, handsVisible: true, commitCount: 2, latestWord: 'more' });
  assert.deepStrictEqual(s.words, ['more', 'more']);
});

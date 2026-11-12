const test = require('node:test');
const assert = require('node:assert');
const loadTs = require('./load-ts.cjs');

const translator = loadTs('src/aslGlossTranslator.ts', {
  './aslSignPoses': loadTs('src/aslSignPoses.ts', {}),
});

test('translates English idioms to ASL glosses correctly', () => {
  const result = translator.translateEnglishToASLGloss('Thank you, you are welcome!');
  assert.match(result.aslSentence, /THANK-YOU/);
  assert.match(result.aslSentence, /WELCOME/);
  assert.ok(result.tokens.length >= 2);
});

test('reorders WH questions to end of sentence in ASL grammar', () => {
  const result = translator.translateEnglishToASLGloss('Where is Room 204?');
  assert.ok(result.aslSentence.endsWith('WHERE'));
  assert.match(result.aslSentence, /ROOM/);
  assert.match(result.aslSentence, /2 0 4/);
});

test('falls back to fingerspelling for unrecognized names/words', () => {
  const result = translator.translateEnglishToASLGloss('Alex');
  const glosses = result.tokens.map((t) => t.gloss);
  assert.deepStrictEqual(glosses, ['A', 'L', 'E', 'X']);
  assert.ok(result.tokens.every((t) => t.type === 'letter'));
});

test('handles empty input gracefully', () => {
  const result = translator.translateEnglishToASLGloss('   ');
  assert.strictEqual(result.aslSentence, '');
  assert.strictEqual(result.tokens.length, 0);
});

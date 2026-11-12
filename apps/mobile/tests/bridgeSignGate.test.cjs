const test = require('node:test');
const assert = require('node:assert');
const gate = require('./load-ts.cjs')('src/bridgeSignGate.ts');

test('a held sign is sent once despite confidence fluctuations', () => {
  let state = gate.initialBridgeSignGate();
  let result = gate.updateBridgeSignGate(state, 1, 'water', 0.84);
  assert.strictEqual(result.shouldSend, true);
  state = result.gate;
  assert.strictEqual(state.armed, false);
  assert.strictEqual(gate.updateBridgeSignGate(state, 1, 'water', 0.2).shouldSend, false);
  assert.strictEqual(gate.updateBridgeSignGate(state, 1, 'water', 0.92).shouldSend, false);
});

test('hands dropping re-arms the same sign for a later gesture', () => {
  let state = gate.initialBridgeSignGate();
  ({ gate: state } = gate.updateBridgeSignGate(state, 1, 'help', 0.9));
  ({ gate: state } = gate.updateBridgeSignGate(state, 0, '', 0));
  assert.strictEqual(gate.updateBridgeSignGate(state, 1, 'help', 0.9).shouldSend, true);
});

test('a different confident sign may be sent without waiting for hands to drop', () => {
  let state = gate.initialBridgeSignGate();
  ({ gate: state } = gate.updateBridgeSignGate(state, 1, 'water', 0.9));
  assert.strictEqual(gate.updateBridgeSignGate(state, 1, 'help', 0.9).shouldSend, true);
});

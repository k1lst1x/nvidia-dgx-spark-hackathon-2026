/**
 * Prevent a held sign from becoming a stream of duplicate Bridge messages.
 * A recognition confidence dip is not evidence that hands were lowered, so it
 * must never re-arm the same word. Only an absent-hand frame does that.
 */
export const BRIDGE_SIGN_SEND_CONFIDENCE = 0.7;

export type BridgeSignGate = {
  armed: boolean;
  lastWord: string;
};

export const initialBridgeSignGate = (): BridgeSignGate => ({ armed: true, lastWord: '' });

export function updateBridgeSignGate(
  gate: BridgeSignGate,
  handCount: number,
  word: string,
  confidence: number,
): { gate: BridgeSignGate; shouldSend: boolean } {
  if (handCount === 0) return { gate: initialBridgeSignGate(), shouldSend: false };
  if (!word || confidence < BRIDGE_SIGN_SEND_CONFIDENCE) return { gate, shouldSend: false };
  if (word === gate.lastWord && !gate.armed) return { gate, shouldSend: false };
  return { gate: { armed: false, lastWord: word }, shouldSend: true };
}

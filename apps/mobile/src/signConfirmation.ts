export type WordGuess = { label: string; confidence: number };

export const WORD_CONFIDENCE_FLOOR = 0.3;
export const WORD_CONFIDENT = 0.45;
/** The model has no "no sign" class, so a near-tie between two words means it is guessing. */
export const WORD_MIN_MARGIN = 0.1;

/** The word one inference window votes for, or '' when it is too unsure to count. */
export function windowVote(top: WordGuess[]): string {
  const [best, second] = top;
  if (!best || best.confidence < WORD_CONFIDENCE_FLOOR) return '';
  if (second && best.confidence - second.confidence < WORD_MIN_MARGIN) return '';
  return best.label;
}

/** Consecutive agreeing windows needed before the word joins the phrase. */
export function requiredStableWindows(confidence: number): number {
  return confidence >= WORD_CONFIDENT ? 2 : 3;
}

import coreVocabularyJson from '../assets/models/asl_core_vocabulary.json';

export type WordGuess = { label: string; confidence: number };
export type SignVocabulary = 'core' | 'all';

export const WORD_CONFIDENCE_FLOOR = 0.3;
export const WORD_CONFIDENT = 0.45;
/** The model has no "no sign" class, so a near-tie between two words means it is guessing. */
export const WORD_MIN_MARGIN = 0.1;
export const CORE_SIGNS: ReadonlySet<string> = new Set((coreVocabularyJson as { signs: string[] }).signs);
export const SIGN_VOCABULARY_KEY = 'guidelens_sign_vocabulary';

/**
 * The 250-word model confidently names words it was never shown, so by
 * default only the core signs can be confirmed (docs/sign/FIXED_VOCABULARY_PLAN.md).
 */
export function readSignVocabulary(): SignVocabulary {
  if (typeof window === 'undefined' || !window.localStorage) return 'core';
  return window.localStorage.getItem(SIGN_VOCABULARY_KEY) === 'all' ? 'all' : 'core';
}

/** The word one inference window votes for, or '' when it is too unsure to count. */
export function windowVote(top: WordGuess[], allowed?: ReadonlySet<string>): string {
  const [best, second] = top;
  if (!best || best.confidence < WORD_CONFIDENCE_FLOOR) return '';
  if (second && best.confidence - second.confidence < WORD_MIN_MARGIN) return '';
  if (allowed && !allowed.has(best.label)) return '';
  return best.label;
}

/** Consecutive agreeing windows needed before the word joins the phrase. */
export function requiredStableWindows(confidence: number): number {
  return confidence >= WORD_CONFIDENT ? 2 : 3;
}

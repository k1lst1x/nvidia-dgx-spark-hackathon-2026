/**
 * Utterance controller for sign recognition.
 *
 * The recognizer confirms one sign at a time. This decides when those signs
 * form a finished utterance, so the app can show provisional text while the
 * person is signing and speak only once, when the utterance is final.
 *
 *   IDLE ──hands seen──▶ SIGNING ──hands down──▶ POSSIBLE_END ──grace over──▶ FINALIZED
 *                           ▲                        │
 *                           └────hands return────────┘
 *
 * A pause is not always the end of a sentence in ASL, so the grace period is
 * deliberately long, hands returning cancels the ending, and the signer can
 * end it themselves with finish().
 *
 * Pure and clock-free: callers pass timestamps in, which makes it testable.
 */

export type UtterancePhase = 'idle' | 'signing' | 'possible_end' | 'finalized';

export type UtteranceOutcome = 'recognized' | 'unrecognized';

export type UtteranceState = {
  phase: UtterancePhase;
  /** Signs confirmed so far in this utterance, in signing order. Provisional until finalized. */
  words: string[];
  /** Set when phase is "finalized". "unrecognized": the person signed but nothing was confirmed. */
  outcome: UtteranceOutcome | null;
  /** Increments each time an utterance is finalized; use it to speak exactly once. */
  finalizedCount: number;
  /** How far through the grace period a possible ending is, 0 to 1. */
  endProgress: number;
};

export type UtteranceInput = {
  now: number;
  handsVisible: boolean;
  /** The recognizer's commit counter; a rise means one more sign was confirmed. */
  commitCount: number;
  /** The sign confirmed most recently (read only when commitCount rises). */
  latestWord?: string;
};

export type UtteranceOptions = {
  /** Hands out of view this long: the utterance may be ending. */
  handsDownMs?: number;
  /** After that, wait this long for the hands to return before finalizing. */
  graceMs?: number;
  /** Hands must have been visible this long for a wordless utterance to count as "unrecognized". */
  minSigningMs?: number;
  /** Longest utterance; older signs are dropped. */
  maxWords?: number;
};

const DEFAULTS: Required<UtteranceOptions> = { handsDownMs: 900, graceMs: 1600, minSigningMs: 1200, maxWords: 16 };

export const initialUtterance: UtteranceState = {
  phase: 'idle', words: [], outcome: null, finalizedCount: 0, endProgress: 0,
};

export class UtteranceController {
  private state: UtteranceState = initialUtterance;
  private readonly options: Required<UtteranceOptions>;
  private lastCommit = 0;
  private startedAt = 0;
  private handsLastSeenAt = 0;
  private signingMs = 0;
  private lastTick = 0;

  constructor(options: UtteranceOptions = {}) {
    this.options = { ...DEFAULTS, ...options };
  }

  get current(): UtteranceState {
    return this.state;
  }

  /** Feed the latest observation. Returns the (possibly unchanged) state. */
  update(input: UtteranceInput): UtteranceState {
    const { now, handsVisible } = input;
    const elapsed = this.lastTick ? Math.max(0, now - this.lastTick) : 0;
    this.lastTick = now;

    // The recognizer was cleared: follow its counter down without inventing a sign.
    if (input.commitCount < this.lastCommit) this.lastCommit = input.commitCount;
    const newWord = input.commitCount > this.lastCommit && input.latestWord ? input.latestWord : null;
    this.lastCommit = Math.max(this.lastCommit, input.commitCount);

    let { phase, words } = this.state;
    const active = handsVisible || newWord !== null;

    if (phase === 'idle' || phase === 'finalized') {
      if (!active) return this.state;
      // A new utterance starts; the previous final text is replaced.
      phase = 'signing';
      words = [];
      this.startedAt = now;
      this.signingMs = 0;
    }
    if (handsVisible) {
      this.handsLastSeenAt = now;
      this.signingMs += elapsed;
    }
    if (newWord) {
      words = [...words, newWord].slice(-this.options.maxWords);
      this.handsLastSeenAt = Math.max(this.handsLastSeenAt, now);
    }

    const handsDownFor = now - this.handsLastSeenAt;
    if (active) {
      phase = 'signing';
    } else if (handsDownFor >= this.options.handsDownMs + this.options.graceMs) {
      return this.finalize(words);
    } else if (handsDownFor >= this.options.handsDownMs) {
      phase = 'possible_end';
    }
    const endProgress = phase === 'possible_end'
      ? Math.min(1, (handsDownFor - this.options.handsDownMs) / this.options.graceMs)
      : 0;
    this.state = { ...this.state, phase, words, outcome: null, endProgress };
    return this.state;
  }

  /** The signer says the sentence is done ("Finish sentence"). */
  finish(): UtteranceState {
    if (this.state.phase === 'idle' || this.state.phase === 'finalized') return this.state;
    return this.finalize(this.state.words);
  }

  /** Forget everything, including the last final text. */
  reset(): UtteranceState {
    this.state = { ...initialUtterance, finalizedCount: this.state.finalizedCount };
    this.signingMs = 0;
    return this.state;
  }

  private finalize(words: string[]): UtteranceState {
    if (!words.length && this.signingMs < this.options.minSigningMs) {
      // Hands passed through the frame briefly: not an attempt to sign.
      this.state = { ...initialUtterance, finalizedCount: this.state.finalizedCount };
      return this.state;
    }
    this.state = {
      phase: 'finalized',
      words,
      outcome: words.length ? 'recognized' : 'unrecognized',
      finalizedCount: this.state.finalizedCount + 1,
      endProgress: 0,
    };
    return this.state;
  }
}

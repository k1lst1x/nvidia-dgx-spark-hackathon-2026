import { resolveOfflinePhrase } from './offlinePhrasebook';

/**
 * Plain English for the compact labels the on-device sign recognizers emit.
 * Labels not listed here are already ordinary English words.
 */
const WORDS: Record<string, string> = {
  callonphone: 'call',
  call: 'call',
  frenchfries: 'French fries',
  glasswindow: 'window',
  haveto: 'have to',
  hesheit: 'he',
  icecream: 'ice cream',
  ily: 'I love you',
  minemy: 'my',
  owie: 'hurt',
  pain: 'hurt',
  shhh: 'quiet',
  thankyou: 'thank you',
  tv: 'TV',
  weus: 'we',
  yourself: 'you',
  done: 'finished',
  finish: 'finished',
};

/** Signs that stand alone as their own sentence rather than joining a clause. */
const STANDALONE: Record<string, string> = {
  hello: 'Hello.',
  bye: 'Goodbye.',
  yes: 'Yes.',
  no: 'No.',
  please: 'Please.',
  thankyou: 'Thank you.',
  okay: 'Okay.',
  sorry: "I'm sorry.",
  ily: 'I love you.',
  peace: 'Peace.',
  help: 'I need help.',
  stop: 'Stop.',
};

const QUESTION_WORDS = new Set(['where', 'who', 'why']);

// ASL has no copula; English needs one between a pronoun and a state.
const COPULA: Record<string, string> = { he: 'is', we: 'are', you: 'are' };
const STATES = new Set([
  'awake', 'bad', 'clean', 'cute', 'dirty', 'dry', 'fine', 'finished', 'happy', 'hot', 'hungry', 'hurt',
  'loud', 'mad', 'noisy', 'old', 'pretty', 'quiet', 'sad', 'safe', 'sick', 'sleepy', 'sticky', 'stuck',
  'thirsty', 'wet',
]);

function key(label: string) {
  return label.trim().toLowerCase().replace(/[\s_-]+/g, '');
}

export function signWordToEnglish(label: string): string {
  const k = key(label);
  if (!k) return '';
  return WORDS[k] ?? label.trim().replace(/[_-]+/g, ' ').toLowerCase();
}

function sentence(words: string[], end: '.' | '?') {
  const text = words.join(' ').trim();
  if (!text) return '';
  return `${text[0].toUpperCase()}${text.slice(1)}${end}`;
}

function clause(labels: string[]): string {
  const words = labels.map(signWordToEnglish).filter(Boolean);
  // ASL usually puts WH-words at the end: BATHROOM WHERE -> Where is bathroom?
  const whIndex = words.findIndex((word) => QUESTION_WORDS.has(word));
  if (whIndex >= 0) {
    const wh = words[whIndex];
    const rest = words.filter((_, index) => index !== whIndex);
    if (rest.length === 0) return sentence([wh], '?');
    return sentence(wh === 'why' ? [wh, ...rest] : [wh, 'is', ...rest], '?');
  }
  const copula = COPULA[words[0]];
  if (copula && words.length === 2 && STATES.has(words[1])) return sentence([words[0], copula, words[1]], '.');
  return sentence(words, '.');
}

/**
 * Compose recognized sign labels, in signing order, into readable English.
 * This only reorders WH-questions, splits greetings, and adds the "is/are"
 * English needs; it never adds content the signer did not sign.
 */
export function glossToEnglish(labels: string[]): string {
  const deduped = labels.filter((label, index) => key(label) && key(label) !== key(labels[index - 1] ?? ''));
  if (deduped.length === 1) {
    const only = deduped[0];
    return STANDALONE[key(only)] ?? resolveOfflinePhrase(key(only))?.speech ?? sentence([signWordToEnglish(only)], '.');
  }
  const sentences: string[] = [];
  let pending: string[] = [];
  for (const label of deduped) {
    const standalone = STANDALONE[key(label)];
    if (standalone) {
      if (pending.length) sentences.push(clause(pending));
      pending = [];
      sentences.push(standalone);
    } else {
      pending.push(label);
    }
  }
  if (pending.length) sentences.push(clause(pending));
  return sentences.join(' ');
}

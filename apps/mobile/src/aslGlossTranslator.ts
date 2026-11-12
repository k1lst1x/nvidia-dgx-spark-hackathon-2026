/**
 * English-to-ASL (American Sign Language) NLP & Gloss Translation Engine.
 *
 * Implements English to ASL grammar transformations:
 * 1. Idiom & multi-word phrase matching ("thank you", "follow me", "how are you")
 * 2. Contraction normalization ("don't" -> "NOT", "I'm" -> "I", "you're" -> "YOU")
 * 3. Removal of English copulas and articles (is, am, are, was, were, the, a, an)
 * 4. WH-question reordering (ASL interrogatives: WHERE, WHAT, WHO placed at sentence end)
 * 5. Sign lexicon mapping with automatic fingerspelling fallback for unrecognized words/names
 */

import {
  ASL_DICTIONARY,
  DIGIT_POSES,
  LETTER_POSES,
  REST_POSE,
  type SignDefinition,
} from './aslSignPoses';

export type ASLToken = {
  id: string;
  gloss: string;
  type: 'word' | 'letter' | 'number' | 'pause';
  text: string;
  sign: SignDefinition;
};

export type ASLTranslationResult = {
  originalText: string;
  aslSentence: string;
  tokens: ASLToken[];
};

// Phrase idioms mapped directly to ASL glosses
const PHRASE_MAPPINGS: Record<string, string[]> = {
  'thank you': ['THANK-YOU'],
  'thanks': ['THANK-YOU'],
  'thank you very much': ['THANK-YOU'],
  "you're welcome": ['WELCOME'],
  'you are welcome': ['WELCOME'],
  'welcome': ['WELCOME'],
  'hello': ['HELLO'],
  'hi': ['HELLO'],
  'hey': ['HELLO'],
  'good morning': ['GOOD', 'HELLO'],
  'good evening': ['GOOD', 'HELLO'],
  'how are you': ['HOW', 'YOU'],
  'i love you': ['I-LOVE-YOU'],
  'i need help': ['I', 'NEED', 'HELP'],
  'help me': ['HELP', 'ME'],
  'please wait': ['PLEASE', 'WAIT'],
  'wait a moment': ['WAIT'],
  'follow me': ['FOLLOW', 'ME'],
  'excuse me': ['EXCUSE-ME'],
  'no problem': ['NO', 'PROBLEM'],
  'goodbye': ['HELLO'],
  'bye': ['HELLO'],
  'see you': ['SEE', 'YOU'],
};

// Words omitted in ASL (copulas, articles, auxiliary fillers)
const ASL_STOPWORDS = new Set([
  'is', 'am', 'are', 'was', 'were', 'been', 'being', 'be',
  'a', 'an', 'the',
  'to', 'of', 'in', 'at', 'on',
  'do', 'does', 'did',
]);

// Contractions map
const CONTRACTIONS: Record<string, string> = {
  "i'm": 'i',
  "i've": 'i',
  "i'll": 'i will',
  "i'd": 'i',
  "you're": 'you',
  "you've": 'you',
  "you'll": 'you will',
  "he's": 'he',
  "she's": 'she',
  "it's": 'it',
  "we're": 'we',
  "they're": 'they',
  "don't": 'not',
  "doesn't": 'not',
  "didn't": 'not',
  "can't": 'cannot',
  "won't": 'will not',
  "what's": 'what',
  "where's": 'where',
  "who's": 'who',
  "how's": 'how',
  "there's": 'there',
};

// WH question words that ASL moves to the end of the clause
const WH_WORDS = new Set(['WHERE', 'WHAT', 'WHO', 'WHEN', 'WHY', 'HOW']);

/**
 * Creates a letter SignDefinition for fingerspelling
 */
function createLetterSign(char: string): SignDefinition {
  const upper = char.toUpperCase();
  const pose = LETTER_POSES[upper] || REST_POSE;
  return {
    gloss: upper,
    englishLabel: `Letter ${upper}`,
    type: 'letter',
    category: 'fingerspelling',
    description: `ASL manual alphabet fingerspelling for '${upper}'`,
    keyframes: [{ landmarks: pose, durationMs: 400 }],
  };
}

/**
 * Creates a digit SignDefinition for numbers
 */
function createDigitSign(digit: string): SignDefinition {
  const pose = DIGIT_POSES[digit] || REST_POSE;
  return {
    gloss: digit,
    englishLabel: `Number ${digit}`,
    type: 'number',
    category: 'numbers',
    description: `ASL manual number for '${digit}'`,
    keyframes: [{ landmarks: pose, durationMs: 420 }],
  };
}

/**
 * Normalizes English text into ASL gloss tokens.
 * Pure and fully functional offline without server dependency.
 */
export function translateEnglishToASLGloss(rawText: string): ASLTranslationResult {
  if (!rawText || !rawText.trim()) {
    return { originalText: '', aslSentence: '', tokens: [] };
  }

  const clean = rawText.trim();
  let normalized = clean.toLowerCase();

  // 1. Expand contractions
  for (const [contraction, expansion] of Object.entries(CONTRACTIONS)) {
    const reg = new RegExp(`\\b${contraction}\\b`, 'gi');
    normalized = normalized.replace(reg, expansion);
  }

  // 2. Extract multi-word phrase idioms
  const glossSequence: Array<{ gloss: string; sourceWord: string; isFingerspell?: boolean }> = [];

  for (const [phrase, glosses] of Object.entries(PHRASE_MAPPINGS)) {
    if (normalized.includes(phrase)) {
      for (const g of glosses) {
        glossSequence.push({ gloss: g, sourceWord: phrase });
      }
      normalized = normalized.replace(phrase, ' ');
    }
  }

  // 3. Process remaining words
  const words = normalized
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);

  let whQuestionWord: string | null = null;

  for (const word of words) {
    if (ASL_STOPWORDS.has(word)) continue;

    const upper = word.toUpperCase();

    // Check if word is a WH-question word
    if (WH_WORDS.has(upper)) {
      whQuestionWord = upper;
      continue;
    }

    // Check if word is in ASL dictionary
    if (ASL_DICTIONARY[upper]) {
      glossSequence.push({ gloss: upper, sourceWord: word });
      continue;
    }

    // Check for synonyms or stem matches
    if (upper === 'HI') {
      glossSequence.push({ gloss: 'HELLO', sourceWord: word });
    } else if (upper === 'THANKS') {
      glossSequence.push({ gloss: 'THANK-YOU', sourceWord: word });
    } else if (upper === 'ASSIST') {
      glossSequence.push({ gloss: 'HELP', sourceWord: word });
    } else if (upper === 'CALL') {
      glossSequence.push({ gloss: 'PHONE', sourceWord: word });
    } else if (upper === 'CORRECT') {
      glossSequence.push({ gloss: 'YES', sourceWord: word });
    } else if (upper === 'NEGATIVE') {
      glossSequence.push({ gloss: 'NO', sourceWord: word });
    } else if (/^\d+$/.test(word)) {
      // Number sequence: spell digit-by-digit
      for (const digit of word) {
        glossSequence.push({ gloss: digit, sourceWord: digit, isFingerspell: true });
      }
    } else {
      // Fingerspell word character by character
      for (const char of word) {
        glossSequence.push({ gloss: char.toUpperCase(), sourceWord: word, isFingerspell: true });
      }
    }
  }

  // 4. WH-question reordering: in ASL, interrogatives go at the end
  if (whQuestionWord) {
    glossSequence.push({ gloss: whQuestionWord, sourceWord: whQuestionWord.toLowerCase() });
  }

  // 5. Convert to executable ASL tokens
  const tokens: ASLToken[] = [];
  let tokenIdx = 0;

  for (const item of glossSequence) {
    const gloss = item.gloss;
    let sign: SignDefinition;

    if (ASL_DICTIONARY[gloss]) {
      sign = ASL_DICTIONARY[gloss];
    } else if (/^[0-9]$/.test(gloss)) {
      sign = createDigitSign(gloss);
    } else if (/^[A-Z]$/.test(gloss)) {
      sign = createLetterSign(gloss);
    } else {
      // Fallback: fingerspell unknown
      sign = createLetterSign(gloss.charAt(0));
    }

    tokens.push({
      id: `token-${tokenIdx++}-${gloss}`,
      gloss,
      type: sign.type,
      text: item.sourceWord,
      sign,
    });
  }

  const aslSentence = tokens.map((t) => t.gloss).join(' ');

  return {
    originalText: clean,
    aslSentence,
    tokens,
  };
}

/**
 * Optional server-assisted translation with transparent offline fallback.
 */
export async function translateEnglishToSign(
  text: string,
  apiUrl?: string,
): Promise<ASLTranslationResult> {
  const local = translateEnglishToASLGloss(text);
  if (!apiUrl) return local;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);
    const response = await fetch(`${apiUrl.replace(/\/$/, '')}/v1/translate-to-sign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) return local;
    const data = await response.json();
    if (data && data.asl_gloss) {
      // Use server gloss ordering while mapping to our rich client poses
      return translateEnglishToASLGloss(data.asl_gloss);
    }
    return local;
  } catch {
    return local;
  }
}

"""English-to-ASL (American Sign Language) Gloss Translation Service.

Transforms English spoken/written grammar into ASL gloss syntax:
- Normalizes idioms and contractions
- Omits English copular verbs ("to be") and articles ("the", "a", "an")
- Reorders interrogative WH-questions to clause ends
- Generates structured token sequences for client-side animated avatar rendering
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

PHRASE_MAPPINGS = {
    "thank you very much": ["THANK-YOU"],
    "thank you": ["THANK-YOU"],
    "thanks": ["THANK-YOU"],
    "you're welcome": ["WELCOME"],
    "you are welcome": ["WELCOME"],
    "how are you": ["HOW", "YOU"],
    "i love you": ["I-LOVE-YOU"],
    "i need help": ["I", "NEED", "HELP"],
    "help me": ["HELP", "ME"],
    "please wait": ["PLEASE", "WAIT"],
    "follow me": ["FOLLOW", "ME"],
    "excuse me": ["EXCUSE-ME"],
    "good morning": ["GOOD", "HELLO"],
}

ASL_STOPWORDS = {
    "is", "am", "are", "was", "were", "been", "being", "be",
    "a", "an", "the",
    "to", "of", "in", "at", "on",
    "do", "does", "did",
}

CONTRACTIONS = {
    r"\bi'm\b": "i",
    r"\bi've\b": "i",
    r"\bi'll\b": "i will",
    r"\byou're\b": "you",
    r"\bhe's\b": "he",
    r"\bshe's\b": "she",
    r"\bit's\b": "it",
    r"\bdon't\b": "not",
    r"\bdoesn't\b": "not",
    r"\bdidn't\b": "not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bwhat's\b": "what",
    r"\bwhere's\b": "where",
}

WH_WORDS = {"WHERE", "WHAT", "WHO", "WHEN", "WHY", "HOW"}

KNOWN_ASL_WORDS = {
    "HELLO", "THANK-YOU", "PLEASE", "HELP", "YES", "NO", "STOP",
    "I-LOVE-YOU", "WATER", "ROOM", "WHERE", "FOLLOW", "WELCOME",
    "PHONE", "WAIT", "GOOD", "NAME", "ME", "YOU", "I", "WE", "THEY",
    "NEED", "WANT", "FOOD", "EAT", "DRINK", "BATHROOM", "DOOR", "CAR",
}


class ASLSignToken(BaseModel):
    gloss: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., description="word, letter, number, or punctuation")
    text: str = Field(..., max_length=128)
    fingerspell: bool = False


class EnglishToSignResponse(BaseModel):
    english: str
    asl_gloss: str
    tokens: list[ASLSignToken]


class EnglishToSignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


def translate_english_to_asl(text: str) -> EnglishToSignResponse:
    clean = text.strip()
    if not clean:
        return EnglishToSignResponse(english="", asl_gloss="", tokens=[])

    normalized = clean.lower()
    for pattern, expansion in CONTRACTIONS.items():
        normalized = re.sub(pattern, expansion, normalized)

    glosses: list[tuple[str, str, str, bool]] = []  # (gloss, type, source_text, fingerspell)

    # 1. Match multi-word phrases
    for phrase, mapped_glosses in PHRASE_MAPPINGS.items():
        if phrase in normalized:
            for g in mapped_glosses:
                glosses.append((g, "word", phrase, False))
            normalized = normalized.replace(phrase, " ")

    # 2. Tokenize words
    tokens = re.findall(r"[a-z0-9]+", normalized)
    wh_word: str | None = None

    for token in tokens:
        if token in ASL_STOPWORDS:
            continue
        upper = token.upper()

        if upper in WH_WORDS:
            wh_word = upper
            continue

        if upper in KNOWN_ASL_WORDS:
            glosses.append((upper, "word", token, False))
        elif token.isdigit():
            for digit in token:
                glosses.append((digit, "number", digit, True))
        else:
            # Word not in core sign dictionary: fingerspell each letter
            for char in upper:
                glosses.append((char, "letter", token, True))

    # 3. In ASL interrogatives, WH words go to the end of the sentence
    if wh_word:
        glosses.append((wh_word, "word", wh_word.lower(), False))

    asl_tokens = [
        ASLSignToken(gloss=g, type=t, text=src, fingerspell=fs)
        for g, t, src, fs in glosses
    ]
    asl_gloss = " ".join(t.gloss for t in asl_tokens)

    return EnglishToSignResponse(
        english=clean,
        asl_gloss=asl_gloss,
        tokens=asl_tokens,
    )

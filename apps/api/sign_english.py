"""Keep sign-to-voice output in readable English."""

import re

_NON_ENGLISH_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_SPACES = re.compile(r"\s+")

# YOLO sign classes that are whole words. Letters and digits are fingerspelling
# fragments and are never spoken on their own.
_YOLO_PHRASES = {
    "ExcuseMe": "Excuse me.",
    "Food": "Food.",
    "Hello": "Hello.",
    "Help": "I need help.",
    "House": "House.",
    "I Love You": "I love you.",
    "Internet": "Internet.",
    "No": "No.",
    "Please": "Please.",
    "ThankYou": "Thank you.",
    "Yes": "Yes.",
}


def english_text(text: str) -> str:
    """Return ``text`` as tidy English, or an empty string when it is not English."""
    cleaned = _SPACES.sub(" ", text or "").strip()
    if not cleaned:
        return ""
    if any(not char.isascii() for char in _NON_ENGLISH_LETTER.findall(cleaned)):
        return ""
    letters = [char for char in cleaned if char.isalpha()]
    if len(letters) > 1 and all(char.isupper() for char in letters):
        cleaned = cleaned.lower()
        cleaned = re.sub(r"\bi\b", "I", cleaned)
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def yolo_sign_phrase(label: str) -> str:
    """Spoken English for a YOLO sign class, or an empty string for letters and digits."""
    return _YOLO_PHRASES.get(label, "")

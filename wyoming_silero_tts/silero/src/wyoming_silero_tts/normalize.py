"""Text normalization for the Russian Silero v5 model.

The model silently drops anything it has no symbol for — digits, Latin
script, emoji — so "Сейчас 13:45" would be spoken as "Сейчас". Numbers are
expanded to Russian words with num2words and Latin words are (optionally)
transliterated to Cyrillic.
"""
from __future__ import annotations

import re

from num2words import num2words

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]+")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DECIMAL = re.compile(r"(\d+)[.,](\d+)")
_INT = re.compile(r"\d+")
_MINUS = re.compile(r"(?:^|(?<=[\s(]))[-−](?=\d)")
# Anything the model has no symbol for becomes a space (after digits and
# Latin have already been rewritten).
_UNSPEAKABLE = re.compile(r"[^а-яёА-ЯЁ\s.,!?…:;()\-—«»\"']")
_SPACES = re.compile(r"\s+")

# Unit symbols worth expanding before number handling ("21,5°C").
_UNITS = [
    ("°C", " градусов Цельсия "),
    ("°F", " градусов Фаренгейта "),
    ("°", " градусов "),
    ("%", " процентов "),
    ("№", " номер "),
]

# Rough Latin-to-Cyrillic transliteration: digraphs first, then single
# letters. Not linguistically perfect — the goal is that "Spotify" is spoken
# recognizably instead of dropped.
_DIGRAPHS = [
    ("shch", "щ"), ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"),
    ("kh", "х"), ("ts", "ц"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"),
    ("ye", "е"), ("ck", "к"), ("th", "т"), ("ph", "ф"), ("oo", "у"),
    ("ee", "и"),
]
_LETTERS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _num(n: int) -> str:
    return num2words(n, lang="ru")


def _expand_time(m: re.Match) -> str:
    hours, minutes = int(m.group(1)), int(m.group(2))
    if hours > 23 or minutes > 59:  # a score like "30:15", not a time
        return f"{_num(hours)} {_num(minutes)}"
    if minutes == 0:
        return f"{_num(hours)} ноль ноль"
    if minutes < 10:
        return f"{_num(hours)} ноль {_num(minutes)}"
    return f"{_num(hours)} {_num(minutes)}"


def _translit_word(m: re.Match) -> str:
    word = m.group(0).lower()
    for latin, cyr in _DIGRAPHS:
        word = word.replace(latin, cyr)
    return "".join(_LETTERS.get(c, c) for c in word)


def normalize(text: str, transliterate: bool = True) -> str:
    """Return speakable text for Silero, or "" if nothing would be voiced."""
    for symbol, replacement in _UNITS:
        text = text.replace(symbol, replacement)
    text = _MINUS.sub("минус ", text)
    text = _TIME.sub(_expand_time, text)
    text = _DECIMAL.sub(lambda m: f"{_num(int(m.group(1)))} и {_num(int(m.group(2)))}", text)
    text = _INT.sub(lambda m: f" {_num(int(m.group(0)))} ", text)
    if transliterate:
        text = _LATIN.sub(_translit_word, text)
    text = _UNSPEAKABLE.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if not _CYRILLIC.search(text):
        return ""
    return text

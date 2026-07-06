"""Vocabulary masks that pin Parakeet TDT decoding to one writing system.

Parakeet TDT v3 has no language conditioning: onnx-asr's ``language=``
argument is silently ignored for transducer models, and the model picks the
language per utterance on its own. On quiet or noisy audio that guess
occasionally lands on English and a Russian command comes out as Latin
gibberish. Since every supported language is written in exactly one script
(Cyrillic, Greek or Latin), banning the other scripts' tokens in the joint
logits forces the decoder back to the configured language's alphabet without
touching the model. Digits, punctuation and special tokens (``<blk>``) are
never banned.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np

_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]")
_CYRILLIC = re.compile(r"[Ѐ-ԯᲀ-ᲈⷠ-ⷿꙀ-ꚟ]")
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")

_SCRIPTS = {"latin": _LATIN, "cyrillic": _CYRILLIC, "greek": _GREEK}

_LANG_SCRIPT = {
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic",
    "el": "greek",
    # Every other Parakeet TDT v3 language writes in Latin script.
}


def load_vocab(vocab_path: str) -> Dict[int, str]:
    """Parse onnx-asr's ``vocab.txt`` ("<piece> <id>" per line)."""
    vocab: Dict[int, str] = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, _, idx = line.rpartition(" ")
            vocab[int(idx)] = piece
    return vocab


def banned_token_ids(vocab: Dict[int, str], language: str) -> Optional[np.ndarray]:
    """Token ids to suppress when decoding ``language``, or None if unknown.

    A token is banned when it contains a letter from a script other than the
    language's own. Script-free tokens (digits, punctuation, the space marker)
    and special tokens like ``<blk>`` are always allowed.
    """
    lang = language.split("-")[0].lower()
    script = _LANG_SCRIPT.get(lang, "latin" if lang else None)
    if script is None:
        return None
    foreign = [p for name, p in _SCRIPTS.items() if name != script]
    banned = [
        idx
        for idx, piece in vocab.items()
        if not (piece.startswith("<") and piece.endswith(">"))
        and any(p.search(piece) for p in foreign)
    ]
    return np.asarray(sorted(banned), dtype=np.int64)

# Changelog

## 1.0.0

- Initial release: Silero `v5_5_ru` (5 Russian voices) over the Wyoming
  protocol, with streaming synthesis (sentence-by-sentence playback).
- Text normalization: numbers, times and decimals are expanded to Russian
  words; Latin words are transliterated to Cyrillic (both would otherwise be
  silently dropped by the model).
- Model download is SHA-256 verified and resumable; torch is installed from
  the official CPU wheel index, all versions pinned.

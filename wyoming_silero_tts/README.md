# Wyoming Silero TTS

Silero v5 Russian text-to-speech served over the Wyoming protocol — a
natural-sounding replacement for Piper's Russian voices in Assist pipelines.

- 5 voices (xenia, baya, kseniya — female; aidar, eugene — male)
- Streaming synthesis: long answers start playing sentence by sentence
- Runs on CPU far faster than real time; no GPU or NPU needed
- Expands numbers to words and transliterates Latin — the Silero model would
  otherwise silently skip them
- Auto-discovered by the Wyoming Protocol integration

Configuration and troubleshooting: see [DOCS.md](./DOCS.md).
Project overview and provenance of all downloads: see the
[repository README](https://github.com/ganiushin/wyoming-parakeet-addon-haos#readme).

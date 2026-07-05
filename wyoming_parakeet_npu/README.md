# Wyoming Parakeet (Intel NPU)

Parakeet TDT 0.6B v3 speech-to-text on the Intel AI Boost NPU, served over the
Wyoming protocol — a fast, low-power replacement for Whisper in Assist voice
pipelines.

- 25 European languages (including Russian and Ukrainian)
- ~200 ms latency for 10 s of audio, ~10–20× less energy than CPU inference
- No on-device model compilation: a precompiled, SHA-256-verified NPU blob is
  downloaded on first start
- Auto-discovered by the Wyoming Protocol integration

Configuration and troubleshooting: see [DOCS.md](./DOCS.md).
Project overview and provenance of all downloads: see the
[repository README](https://github.com/ganiushin/parakeet-stt-silero-tts-addons-haos#readme).

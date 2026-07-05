# Wyoming Parakeet (Intel NPU)

Parakeet TDT 0.6B v3 speech-to-text on Intel NPUs (Core Ultra AI Boost), exposed over the Wyoming protocol — a fast, low-power drop-in replacement for Whisper in Assist voice pipelines.

- 25 European languages (including Russian and Ukrainian)
- ~200 ms latency for 10 s of audio on the NPU (vs ~1 s on CPU), ~10–20× less energy per inference
- Auto-discovered by the Wyoming Protocol integration

See [DOCS.md](./DOCS.md) for configuration.

Built entirely from source vendored in this repository (`parakeet/`), originally from [cibernox/wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu) (MIT).

# Wyoming Parakeet on Intel NPU — Home Assistant add-on repository

Home Assistant OS add-on: NVIDIA Parakeet TDT 0.6B v3 speech-to-text accelerated on Intel NPUs (Core Ultra / AI Boost) via OpenVINO, exposed over the Wyoming protocol.

The application source code is taken from the original project [cibernox/wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu) by Miguel Camba (MIT license) and vendored into this repository (`wyoming_parakeet_npu/parakeet/`), so the add-on image is built entirely from the code kept here — no prebuilt third-party images. See [VENDORED.md](./wyoming_parakeet_npu/parakeet/VENDORED.md) for the exact upstream commit.

A drop-in replacement for the Whisper add-on in Assist voice pipelines: 25 European languages, ~200 ms latency for 10 s of audio on an NPU, a fraction of the CPU power draw.

## Installation

[![Add repository to my Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fganiushin%2Fwyoming-parakeet-addon-haos)

Or manually: **Settings → Add-ons → Add-on Store → ⋮ → Repositories** and add:

```
https://github.com/ganiushin/wyoming-parakeet-addon-haos
```

Then install **Wyoming Parakeet (Intel NPU)** from the store.

## Requirements

- x86-64 machine with an Intel Core Ultra CPU (Meteor Lake / Lunar Lake / Arrow Lake) with AI Boost NPU
- Home Assistant OS with a kernel that provides the `intel_vpu` driver — check that `/dev/accel/accel0` exists on the host
- ~4 GB free disk space (models are downloaded on first start)

## Add-ons

| Add-on | Description |
|---|---|
| [Wyoming Parakeet (Intel NPU)](./wyoming_parakeet_npu) | Parakeet TDT 0.6B v3 STT on Intel NPU, Wyoming protocol server |

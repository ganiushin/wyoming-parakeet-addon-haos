# Wyoming Parakeet on Intel NPU

**Local speech-to-text for Home Assistant that runs on the Intel NPU — fast, multilingual, and fully self-hosted.**

[![Home Assistant Add-on](https://img.shields.io/badge/Home%20Assistant-add--on-41BDF5?logo=homeassistant&logoColor=white)](https://www.home-assistant.io/addons/)
[![Architecture](https://img.shields.io/badge/arch-amd64-informational)](#requirements)
[![Model](https://img.shields.io/badge/model-Parakeet%20TDT%200.6B%20v3-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
[![License](https://img.shields.io/badge/add--on-MIT-green)](./wyoming_parakeet_npu/parakeet/LICENSE)

A Home Assistant OS add-on that serves NVIDIA **Parakeet TDT 0.6B v3** over the
[Wyoming protocol](https://github.com/rhasspy/wyoming), with inference compiled
for the **Intel AI Boost NPU** (Core Ultra) via OpenVINO. A drop-in replacement
for the Whisper add-on in Assist voice pipelines.

## Why

| | Whisper (CPU) | **Parakeet (NPU)** |
|---|---|---|
| Latency, 10 s of audio | ~1 s | **~200 ms** |
| Energy per request | ~45 J | **~4 J** |
| Languages | many | 25 European (incl. Russian, Ukrainian) |
| Extra hardware | separate server / GPU | **the NPU already in your CPU** |

First start is instant-ish too: instead of compiling the model on your machine
(a multi-GB memory peak), the add-on downloads a **precompiled, SHA-256-verified
NPU blob** and simply loads it.

## Installation

[![Add repository to my Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fganiushin%2Fwyoming-parakeet-addon-haos)

Or manually: **Settings → Add-ons → Add-on Store → ⋮ → Repositories** →
`https://github.com/ganiushin/wyoming-parakeet-addon-haos`, then install
**Wyoming Parakeet (Intel NPU)**.

After the first start the **Wyoming Protocol** integration is discovered
automatically — accept it under **Settings → Devices & Services**, then select
the engine in **Settings → Voice assistants**.

## Requirements

- **CPU:** Intel Core Ultra with AI Boost NPU — Meteor Lake, Arrow Lake (tested) or Lunar Lake
- **Host:** `/dev/accel/accel0` present (kernel ≥ 6.10 with `intel_vpu`)
- **Proxmox VM:** pass the NPU through as a PCI device **and** add `intel_vpu.force_snoop=1` to `/mnt/boot/cmdline.txt` in HAOS ([details](./wyoming_parakeet_npu/DOCS.md#running-haos-in-a-proxmox-vm))
- **Memory:** ~2 GB free for the add-on — on Proxmox give the HAOS VM **4 GB**
- **Disk:** ~6 GB in the add-on data volume (model files + NPU blob)

## Everything this add-on downloads

The image is built **entirely from source contained in this repository** — no
prebuilt third-party images. These are the only external artifacts, all pinned:

| Artifact | Source | Verification |
|---|---|---|
| `ubuntu:24.04` base image | `mirror.gcr.io/library/ubuntu` (Google mirror of Docker Hub) | image digest resolved at build |
| Intel NPU driver + compiler v1.33.0 | [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases/tag/v1.33.0) | **SHA-256 pinned** in Dockerfile |
| oneAPI Level Zero loader 1.28.6 | [oneapi-src/level-zero](https://github.com/oneapi-src/level-zero/releases/tag/v1.28.6) | **SHA-256 pinned** in Dockerfile |
| Python packages (`openvino==2026.2.1`, `onnxruntime==1.26.0`, `wyoming==1.7.2`, …) | PyPI | versions pinned in [`pyproject.toml`](./wyoming_parakeet_npu/parakeet/pyproject.toml) |
| Parakeet model files (~3.2 GB, ONNX) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) on Hugging Face | downloaded at first start |
| Precompiled NPU blob (10 s bucket, ~1.2 GB) | [this repo's `blobs-1` release](https://github.com/ganiushin/wyoming-parakeet-addon-haos/releases/tag/blobs-1) | **SHA-256 pinned** in [`blobs.json`](./wyoming_parakeet_npu/blobs.json) |

The application source lives in
[`wyoming_parakeet_npu/parakeet/`](./wyoming_parakeet_npu/parakeet/) — vendored
from [cibernox/wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu)
(MIT, see [VENDORED.md](./wyoming_parakeet_npu/parakeet/VENDORED.md) for the exact commit).

## How it works

```
Assist pipeline ──Wyoming──▶ add-on :10300
                              │
                              ▼
                 onnx-asr pipeline (preprocess, TDT decoding)
                              │
              encoder ────────┴──────── decoder/joint
                 │                          │
   precompiled NPU blob             static-shape OpenVINO IR
   (mmap-imported, no             (compiled on device, cached)
    on-device compile)
                 └──────────▶ Intel NPU ◀──┘
```

The NPU compiler requires static input shapes, so the encoder is compiled for a
fixed audio length ("bucket", default 10 s; shorter audio is padded, longer is
truncated). The 10 s bucket ships precompiled; other bucket sizes are compiled
on the device once and cached. See [the add-on docs](./wyoming_parakeet_npu/DOCS.md)
for options and troubleshooting.

## Credits

- [cibernox/wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu) — the original project this add-on packages (MIT, © Miguel Camba)
- [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) and the [ONNX export](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) of the model (MIT / CC-BY-4.0)
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — the model itself (CC-BY-4.0)

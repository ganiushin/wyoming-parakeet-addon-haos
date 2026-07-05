# Wyoming Parakeet (Intel NPU)

Runs [wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu): NVIDIA Parakeet TDT 0.6B v3 multilingual speech-to-text, accelerated on Intel NPUs (AI Boost) via OpenVINO and exposed over the Wyoming protocol. Use it as the STT engine in Assist voice pipelines instead of Whisper.

## Requirements

- Intel Core Ultra CPU with an AI Boost NPU (Arrow Lake verified upstream; Meteor Lake and Lunar Lake should work).
- `/dev/accel/accel0` must exist on the Home Assistant OS host (`intel_vpu` kernel driver). Check from the SSH add-on with protection mode off: `ls /dev/accel/`.
- ~4 GB free disk space. On first start the add-on downloads ~3.2 GB of model files into its persistent data directory; later starts take seconds.

## First start

The first start is slow: models are downloaded and compiled for the NPU. Watch the add-on log. Once the Wyoming server is listening, the add-on registers itself with Home Assistant and the **Wyoming Protocol** integration is offered under **Settings → Devices & Services** (accept it, or add it manually with the host IP and port `10300`).

Then select the new STT engine in **Settings → Voice assistants** for your pipeline.

## Options

### `language`

Default transcription language, used when the pipeline does not specify one. Parakeet TDT 0.6B v3 supports 25 European languages: `bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro ru sk sl es sv uk`.

### `device`

OpenVINO device to run inference on: `NPU` (default), `GPU`, or `CPU`. `CPU` is a useful fallback to verify the pipeline works if the NPU is not detected. `GPU` requires the host to expose `/dev/dri` to the add-on, which this add-on does not map — use `NPU` or `CPU`.

### `encoder_buckets` / `encoder_lazy_buckets`

Comma-separated audio bucket sizes in seconds. Eager buckets are prepared at startup; lazy buckets on first use. Audio longer than the largest bucket is truncated.

For bucket sizes **6, 7, 8, 9 or 10 seconds** the add-on downloads a prebuilt, SHA-256-verified NPU blob (compiled offline for NPU 3720 — Meteor/Arrow Lake) instead of compiling on the device, so even the first start is light on memory (~1.5 GB peak). Any other size falls back to on-device compilation, which peaks at several GB of RAM once (the resulting blob is then cached in `/data/ov_cache`). Recommended on small VMs: `encoder_buckets: "6"` (or up to `10`), `encoder_lazy_buckets` empty.

## Notes and limitations

- No streaming transcription: the whole utterance is transcribed at once (normal for Assist pipelines).
- amd64 only — the Intel NPU is x86 by definition.
- The image is built locally from source code vendored in this repository (`parakeet/`, see `parakeet/VENDORED.md` for provenance); the first install therefore takes several minutes while Python dependencies and the Intel NPU driver packages (SHA-256 verified) are installed.

## Troubleshooting

- **`/dev/accel/accel0 not found` in the log** — the host kernel does not have `intel_vpu` loaded or the NPU is unsupported. Set `device: CPU` to test the rest of the pipeline.
- **Discovery did not appear** — add the Wyoming integration manually: **Settings → Devices & Services → Add integration → Wyoming Protocol**, host = your HA machine IP, port = `10300`.

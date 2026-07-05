# Wyoming Parakeet (Intel NPU)

Runs [wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu): NVIDIA Parakeet TDT 0.6B v3 multilingual speech-to-text, accelerated on Intel NPUs (AI Boost) via OpenVINO and exposed over the Wyoming protocol. Use it as the STT engine in Assist voice pipelines instead of Whisper.

## Requirements

- Intel Core Ultra CPU with an AI Boost NPU (Arrow Lake verified; Meteor Lake and Lunar Lake should work).
- `/dev/accel/accel0` must exist on the Home Assistant OS host (`intel_vpu` kernel driver). Check from the SSH add-on with protection mode off: `ls /dev/accel/`.
- ~6 GB free disk space in the add-on data volume (model files + NPU blob).
- ~2 GB of free RAM for the add-on. On Proxmox with NPU passthrough give the HAOS VM at least 5 GB — passthrough pins the guest RAM, and a 4 GB VM running a typical add-on set OOM-kills the model load (exit code 137).

## First start

On first start the add-on downloads ~3.2 GB of model files plus a ~1.2 GB precompiled NPU blob (SHA-256 verified) into its persistent data directory — watch the progress in the add-on log. No on-device model compilation happens for the default 10 s bucket. Later starts take seconds.

Once the Wyoming server is listening, the add-on registers itself with Home Assistant and the **Wyoming Protocol** integration is offered under **Settings → Devices & Services** (accept it, or add it manually with the host IP and port `10300`).

Then select the new STT engine in **Settings → Voice assistants** for your pipeline.

## Options

### `language`

Default transcription language, used when the pipeline does not specify one. Parakeet TDT 0.6B v3 supports 25 European languages: `bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro ru sk sl es sv uk`.

### `device`

OpenVINO device to run inference on: `NPU` (default), `GPU`, or `CPU`. `CPU` is a useful fallback to verify the pipeline works if the NPU is not detected. `GPU` requires the host to expose `/dev/dri` to the add-on, which this add-on does not map — use `NPU` or `CPU`.

### `encoder_buckets` / `encoder_lazy_buckets`

Comma-separated audio bucket sizes in seconds. Eager buckets are prepared at startup; lazy buckets on first use. Audio shorter than the bucket is padded (and still costs a full-bucket inference — ~200 ms on the NPU for 10 s); audio longer than the largest bucket is truncated.

The default **10 s** bucket uses a prebuilt, SHA-256-verified NPU blob (compiled offline for NPU 3720 — Meteor/Arrow Lake) downloaded on first start, so no on-device compilation is needed. Any other size falls back to a one-time on-device compile, which peaks at **~5 GB of RAM** (the resulting blob is then cached in `/data/ov_cache` and later starts are cheap again). Each configured eager bucket keeps its own ~1.2 GB compiled copy in memory — on small VMs stick to a single bucket.

## Notes and limitations

- No streaming transcription: the whole utterance is transcribed at once (normal for Assist pipelines).
- amd64 only — the Intel NPU is x86 by definition.
- The image is built locally from source code vendored in this repository (`parakeet/`, see `parakeet/VENDORED.md` for provenance); the first install therefore takes several minutes while Python dependencies and the Intel NPU driver packages (SHA-256 verified) are installed.

## Troubleshooting

- **`/dev/accel/accel0 not found` in the log** — the host kernel does not have `intel_vpu` loaded or the NPU is unsupported. Set `device: CPU` to test the rest of the pipeline.
- **Discovery did not appear** — add the Wyoming integration manually: **Settings → Devices & Services → Add integration → Wyoming Protocol**, host = your HA machine IP, port = `10300`.

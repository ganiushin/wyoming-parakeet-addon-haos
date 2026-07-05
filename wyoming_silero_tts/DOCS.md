# Wyoming Silero TTS

Runs [Silero](https://github.com/snakers4/silero-models) `v5_5_ru` Russian
text-to-speech on the CPU and exposes it over the Wyoming protocol. Use it as
the TTS engine in Assist voice pipelines instead of Piper — the voices are
markedly more natural than Piper's Russian ones.

## Requirements

- Any amd64 or aarch64 machine; no GPU or NPU needed. Synthesis runs ~50–100×
  faster than real time on two CPU threads.
- ~150 MB free disk space in the add-on data volume (the model file).
- ~1 GB of free RAM for the add-on (~750 MB resident after warm-up).

## First start

On first start the add-on downloads the model package (~139 MB, SHA-256
verified, resumable) into its persistent data directory. Later starts take
seconds; a short warm-up synthesis runs before the port opens.

Once the Wyoming server is listening, the add-on registers itself with Home
Assistant and the **Wyoming Protocol** integration is offered under
**Settings → Devices & Services** (accept it, or add it manually with the
host IP and port `10200`).

Then select the new TTS engine and voice in **Settings → Voice assistants**
for your pipeline.

> **Port conflict with Piper:** this add-on uses host port `10200`, the same
> as the official Piper add-on. Stop Piper first, or change the host port on
> this add-on's Configuration → Network panel if you want to run both.

## Options

### `voice`

Default speaker, used when the pipeline does not specify one:

| voice | |
|---|---|
| `xenia` | female, neutral (default) |
| `baya` | female, soft |
| `kseniya` | female, bright |
| `aidar` | male, neutral |
| `eugene` | male, low |

All five are always installed; the pipeline can pick any of them per request.

### `sample_rate`

Output sample rate: `48000` (default, best quality), `24000` or `8000`.

### `threads`

Torch CPU threads for synthesis (default `2`). Two threads already
synthesize far faster than real time; raise only if responses feel slow on a
very weak CPU.

### `transliterate`

The Russian Silero model silently skips Latin script. When enabled (default),
Latin words — device names, "Wi-Fi", "Spotify" — are transliterated to
Cyrillic so they are spoken instead of dropped. The transliteration is
letter-based and approximate; disable it if you prefer Latin words silent.

## Text normalization

The model also drops bare digits, so the add-on expands them before
synthesis: integers and decimals become Russian words (`21,5` → «двадцать
один и пять»), times are read as hours and minutes (`13:45` → «тринадцать
сорок пять»), and `%`, `°C`, `°F`, `№` are spelled out.

## Model license

The add-on code is MIT. The Silero model weights are distributed by
[snakers4/silero-models](https://github.com/snakers4/silero-models) under
**CC BY-NC-SA 4.0** — free for personal, non-commercial use, which is what a
home Assist pipeline is. Commercial deployments need a license from Silero.

## Troubleshooting

- **The add-on speaks numbers oddly.** Russian number agreement (gender and
  case) is approximated; «двадцать один градус» comes out fine, some
  combinations less so. Open an issue with the exact phrase.
- **A sentence is skipped entirely.** After normalization nothing speakable
  remained (e.g. only emoji or punctuation) — this is by design, the stream
  continues with the next sentence.
- **First response after a restart is slow.** The first synthesis pays
  one-time initialization; the add-on warms up at startup, but if you query
  it during the model download/load window the reply waits for that.

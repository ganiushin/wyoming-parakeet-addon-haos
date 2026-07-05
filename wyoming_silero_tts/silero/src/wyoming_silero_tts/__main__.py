"""Wyoming server for Silero v5 Russian text-to-speech on CPU.

Loads the torch.package model downloaded by scripts/bootstrap.py and serves
it over the Wyoming protocol with streaming synthesis support.

Designed for one specific use case: natural Russian voices for Home
Assistant Assist pipelines on modest x86/ARM CPUs. No model selection —
just Silero v5 (v5_5_ru).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from functools import partial

import torch
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .handler import SileroEventHandler

_LOGGER = logging.getLogger(__name__)

MODEL_FILE = "v5_5_ru.pt"

_VOICE_DESCRIPTIONS = {
    "xenia": "Female, neutral",
    "baya": "Female, soft",
    "kseniya": "Female, bright",
    "aidar": "Male, neutral",
    "eugene": "Male, low",
}


def _build_wyoming_info(speakers: list) -> Info:
    silero_attribution = Attribution(
        name="Silero (snakers4/silero-models)",
        url="https://github.com/snakers4/silero-models",
    )
    return Info(tts=[TtsProgram(
        name="silero",
        description="Silero v5 Russian text-to-speech on CPU",
        attribution=Attribution(
            name="ganiushin/wyoming-parakeet-addon-haos",
            url="https://github.com/ganiushin/wyoming-parakeet-addon-haos",
        ),
        installed=True,
        version=__version__,
        supports_synthesize_streaming=True,
        voices=[
            TtsVoice(
                name=speaker,
                description=_VOICE_DESCRIPTIONS.get(speaker),
                attribution=silero_attribution,
                installed=True,
                version="v5.5",
                languages=["ru"],
            )
            for speaker in speakers
        ],
    )])


def _env_flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wyoming-silero-tts",
        description="Wyoming TTS server for Silero v5 Russian voices",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10200"),
        help="Wyoming server URI (default: tcp://0.0.0.0:10200)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "/data"),
        help="Directory holding the model file (default: /data)",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("VOICE", "xenia"),
        help="Default speaker when the client does not specify one",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.environ.get("SAMPLE_RATE", "48000")),
        choices=[8000, 24000, 48000],
        help="Output sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("THREADS", "2")),
        help="Torch CPU threads for synthesis (default: 2)",
    )
    parser.add_argument(
        "--no-transliterate",
        action="store_true",
        default=not _env_flag("TRANSLITERATE"),
        help="Do not transliterate Latin words to Cyrillic",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("wyoming-silero-tts v%s", __version__)
    torch.set_num_threads(max(1, args.threads))

    model_path = os.path.join(args.data_dir, "silero", MODEL_FILE)
    if not os.path.exists(model_path):
        _LOGGER.error(
            "Model not found at %s. The entrypoint should fetch it on first "
            "run; if you bypassed the entrypoint, run scripts/bootstrap.py "
            "manually.",
            model_path,
        )
        sys.exit(1)

    _LOGGER.info("Loading Silero model from %s ...", model_path)
    t0 = time.perf_counter()
    importer = torch.package.PackageImporter(model_path)
    model = importer.load_pickle("tts_models", "model")
    model.to("cpu")
    _LOGGER.info("Model loaded in %.1f s; speakers: %s",
                 time.perf_counter() - t0, model.speakers)

    voice = args.voice
    if voice not in model.speakers:
        _LOGGER.warning("Voice %r not in model speakers; using %s",
                        voice, model.speakers[0])
        voice = model.speakers[0]

    # First apply_tts call pays one-time lazy-init costs (~10x a normal
    # request); warm up now so the first real request is instant. Doubles as
    # a self-test that the model actually synthesizes.
    t0 = time.perf_counter()
    model.apply_tts(text="Голосовой сервер запущен.", speaker=voice,
                    sample_rate=args.sample_rate)
    _LOGGER.info("Warm-up synthesis took %.2f s", time.perf_counter() - t0)

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    info = _build_wyoming_info(list(model.speakers))
    _LOGGER.info("Ready. Listening on %s (voice=%s rate=%d)",
                 args.uri, voice, args.sample_rate)
    await server.run(partial(
        SileroEventHandler, info, model, model_lock,
        voice=voice,
        sample_rate=args.sample_rate,
        transliterate=not args.no_transliterate,
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

"""Bootstrap helper: download the Silero TTS model package.

Run once on first start (from docker-entrypoint.sh) to populate /data.
Idempotent — a present, size-nonzero model file is trusted (it was SHA-256
verified when first downloaded; re-hashing 139 MB on every start is wasted
startup time on the low-power boxes this add-on targets).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

MODEL_FILE = "v5_5_ru.pt"
MODEL_URL = f"https://models.silero.ai/models/tts/ru/{MODEL_FILE}"
# Pinned digest of the upstream package (2026-07-05) so a silently re-published
# model can't slip in.
MODEL_SHA256 = "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437"


def _fetch_resumable(url: str, tmp: Path, attempts: int = 3) -> None:
    """Download to ``tmp``, resuming a partial file via HTTP Range."""
    import urllib.request

    for attempt in range(1, attempts + 1):
        pos = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if pos:
            req.add_header("Range", f"bytes={pos}-")
            print(f"[bootstrap]   resuming from {pos / 1e6:.1f} MB", flush=True)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if pos and resp.status != 206:
                    pos = 0  # server ignored Range; start over
                with open(tmp, "ab" if pos else "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001 — retried below
            if attempt == attempts:
                raise
            print(f"[bootstrap]   download interrupted ({exc}); retrying "
                  f"({attempt}/{attempts})", file=sys.stderr)
            time.sleep(5)


def download_model(model_dir: Path) -> None:
    target = model_dir / MODEL_FILE
    if target.exists() and target.stat().st_size > 0:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    # Two full tries: a checksum mismatch discards the file and re-downloads
    # once; a second mismatch is a hard error (the model is mandatory).
    for attempt in (1, 2):
        print(f"[bootstrap] Downloading {MODEL_FILE} ...", flush=True)
        t0 = time.perf_counter()
        _fetch_resumable(MODEL_URL, tmp)
        digest = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() == MODEL_SHA256:
            tmp.rename(target)
            print(
                f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
                f"({target.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )
            return
        print(f"[bootstrap]   checksum mismatch for {MODEL_FILE}; "
              f"discarding (attempt {attempt}/2)", file=sys.stderr)
        tmp.unlink()

    print("[bootstrap] FATAL: model download failed checksum verification twice",
          file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/data"))
    args = ap.parse_args()

    download_model(Path(args.data_dir) / "silero")
    print("[bootstrap] All assets ready.", flush=True)


if __name__ == "__main__":
    main()

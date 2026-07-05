"""Bootstrap helper: download Parakeet model files and build the static decoder IR.

Run once on first start (from docker-entrypoint.sh) to populate /data with
everything the server needs. Idempotent — skips files that already exist.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"

# Optional manifest with prebuilt NPU encoder blobs (shipped by the add-on).
BLOB_MANIFEST = Path("/app/blobs.json")

# Files we need locally. Some are pulled by onnx-asr's HF auto-download for the
# INT8 pipeline init, but we fetch all of them explicitly so the first run is
# deterministic and we don't depend on onnx-asr's download path resolving.
REQUIRED_FILES = [
    "config.json",
    "vocab.txt",
    "nemo128.onnx",                  # Mel preprocessor (~140 KB)
    "encoder-model.int8.onnx",       # INT8 encoder for onnx_asr pipeline init (~650 MB)
    "decoder_joint-model.int8.onnx", # INT8 decoder for onnx_asr pipeline init (~18 MB)
    "encoder-model.onnx",            # FP32 encoder graph (~41 MB)
    "encoder-model.onnx.data",       # FP32 encoder weights (~2.4 GB external)
    "decoder_joint-model.onnx",      # FP32 decoder/joint (~72 MB)
]


def download_if_missing(model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    for fname in REQUIRED_FILES:
        target = model_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"[bootstrap] Downloading {fname} from {HF_REPO} ...", flush=True)
        t0 = time.perf_counter()
        hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            local_dir=str(model_dir),
        )
        print(
            f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
            f"({target.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )


def _wanted_blob_files() -> set:
    """Blob filenames the current bucket/device configuration would use."""
    device = os.environ.get("DEVICE", "NPU")
    ts = set()
    for var in ("ENCODER_BUCKETS", "ENCODER_LAZY_BUCKETS"):
        for x in os.environ.get(var, "").split(","):
            x = x.strip()
            if x:
                ts.add(int(float(x) * 100))
    return {f"encoder_T{t}_{device}.blob" for t in ts}


def download_prebuilt_blobs(cache_dir: Path) -> None:
    """Fetch prebuilt NPU blobs from the manifest so no on-device compile
    (with its multi-GB memory peak) is ever needed for the listed buckets."""
    import hashlib
    import json
    import urllib.request

    if not BLOB_MANIFEST.exists():
        return
    wanted = _wanted_blob_files()
    for entry in json.loads(BLOB_MANIFEST.read_text()).get("encoder_blobs", []):
        fname = entry["file"]
        if fname not in wanted:
            continue
        target = cache_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        print(f"[bootstrap] Downloading precompiled NPU blob {fname} ...", flush=True)
        try:
            t0 = time.perf_counter()
            urllib.request.urlretrieve(entry["url"], tmp)
            digest = hashlib.sha256()
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                print(f"[bootstrap]   checksum mismatch for {fname}; "
                      "discarding (will compile on device instead)", file=sys.stderr)
                tmp.unlink()
                continue
            tmp.rename(target)
            print(
                f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
                f"({target.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — blob prefetch is best-effort
            print(f"[bootstrap]   blob download failed: {exc} "
                  "(will compile on device instead)", file=sys.stderr)
            if tmp.exists():
                tmp.unlink()


def build_static_decoder_ir(model_dir: Path, out_dir: Path) -> None:
    """Static-reshape the FP32 decoder/joint to fixed shapes and save as OV IR."""
    out_xml = out_dir / "decoder-static.xml"
    if out_xml.exists() and out_xml.stat().st_size > 0:
        print(f"[bootstrap] Static decoder IR exists at {out_xml}; skipping", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[bootstrap] Building static decoder IR ...", flush=True)
    import openvino as ov  # imported here so docker layer ordering doesn't matter

    src = model_dir / "decoder_joint-model.onnx"
    if not src.exists():
        print(f"[bootstrap] FATAL: {src} not found", file=sys.stderr)
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(str(src))
    # Static shapes match the per-token decoder loop exactly: one frame of
    # encoder output, one previous token, two LSTM states.
    model.reshape({
        "encoder_outputs": [1, 1024, 1],
        "targets": [1, 1],
        "target_length": [1],
        "input_states_1": [2, 1, 640],
        "input_states_2": [2, 1, 640],
    })
    ov.save_model(model, str(out_xml))
    print(f"[bootstrap]   saved {out_xml}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = data_dir / "nemo-parakeet-tdt-0.6b-v3"
    decoder_dir = data_dir / "static_decoder"

    download_if_missing(model_dir)
    build_static_decoder_ir(model_dir, decoder_dir)
    download_prebuilt_blobs(data_dir / "ov_cache")
    print("[bootstrap] All assets ready.", flush=True)


if __name__ == "__main__":
    main()

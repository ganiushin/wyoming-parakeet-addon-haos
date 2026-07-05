"""OpenVINO NPU shims that replace onnx_asr's encoder + decoder ORT sessions.

Why: the Parakeet encoder requires static input shapes for NPU compilation, and
ORT's CPUExecutionProvider can't talk to the NPU. We compile static-shape buckets
of the encoder (and one decoder/joint) for OpenVINO NPU and inject them into the
loaded `onnx_asr` model in place of its default ORT InferenceSessions.

Wiring note: the shim must be assigned to `model.asr._encoder` and
`model.asr._decoder_joint` (NOT `model._encoder`). `model` is a
TextResultsAsrAdapter that wraps the actual ASR object on `.asr`; the wrapper
does not proxy attribute writes.
"""
from __future__ import annotations

import io
import logging
import mmap
import os
from typing import Optional

import numpy as np
import openvino as ov

_LOGGER = logging.getLogger(__name__)


class _MmapBlob(io.BytesIO):
    """BytesIO-typed wrapper whose buffer is an mmap of the blob file.

    ov.Core.import_model() only accepts bytes/BytesIO and grabs the raw
    buffer via getbuffer(). Backing that buffer with mmap lets the kernel
    page the file in on demand instead of holding a full in-memory copy —
    on a 1.2 GB blob this halves the import memory peak.
    """

    def __init__(self, path: str):
        super().__init__()
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, prot=mmap.PROT_READ)

    def getbuffer(self):
        return memoryview(self._mm)

    def close(self):
        self._mm.close()
        self._f.close()
        super().close()


def _build_compile_cfg(device: str, cache_dir: str) -> dict:
    cfg = {"CACHE_DIR": cache_dir, "PERFORMANCE_HINT": "LATENCY"}
    if device == "NPU":
        cfg["NPU_TURBO"] = "YES"
    return cfg


class _Spec:
    """Mimics ORT's NodeArg interface for ``get_inputs()`` / ``get_outputs()``."""

    def __init__(self, name, shape):
        self.name = name
        self.shape = list(shape)


class _SingleEncoder:
    """One static-shape compiled encoder."""

    def __init__(self, compiled, T_fixed: int):
        self._compiled = compiled
        self._req = self._compiled.create_infer_request()
        self.T = T_fixed
        self.MEL = 128
        self.B = 1
        self._inputs = list(self._compiled.inputs)
        self._outputs = list(self._compiled.outputs)
        self._out_names = [o.any_name for o in self._outputs]

    def infer(self, audio: np.ndarray, real_length: int) -> dict:
        b, m, t = audio.shape
        if t < self.T:
            padded = np.zeros((self.B, self.MEL, self.T), dtype=np.float32)
            padded[:, :, :t] = audio
        elif t > self.T:
            padded = audio[:, :, : self.T].astype(np.float32, copy=False)
        else:
            padded = audio.astype(np.float32, copy=False)
        clamped = np.array([min(real_length, self.T)], dtype=np.int64)
        result = self._req.infer({"audio_signal": padded, "length": clamped})
        return {k.any_name: np.array(v) for k, v in result.items()}


class OpenVinoEncoderShim:
    """Encoder shim with multi-bucket dispatch and optional lazy loading.

    Eager buckets compile at construction time and stay hot. Lazy buckets are
    placeholders until first request, then compile (cache-hit on warm restart)
    and stay hot. Useful for rarely-hit large buckets to keep startup memory
    bounded without sacrificing latency on the common path.
    """

    def __init__(
        self,
        *,
        onnx_path: str,
        device: str,
        cache_dir: str,
        eager_seconds: list[float],
        lazy_seconds: Optional[list[float]] = None,
        fps: int = 100,
    ):
        if not eager_seconds:
            raise ValueError("At least one eager bucket is required")
        self._onnx_path = onnx_path
        self._device = device
        self._cache_dir = cache_dir
        self._buckets: dict[int, Optional[_SingleEncoder]] = {}

        for sec in eager_seconds:
            T = int(sec * fps)
            self._buckets[T] = self._build(T)
        for sec in (lazy_seconds or []):
            T = int(sec * fps)
            self._buckets.setdefault(T, None)

        self._sorted_Ts = sorted(self._buckets.keys())
        first = next(b for b in self._buckets.values() if b is not None)
        self._inputs_meta = first._inputs
        self._outputs_meta = first._outputs

        eager = sorted([T for T, b in self._buckets.items() if b is not None])
        lazy = sorted([T for T, b in self._buckets.items() if b is None])
        _LOGGER.info(
            "[encoder] %d bucket(s) total | eager: %s | lazy: %s",
            len(self._buckets), eager, lazy,
        )
        self._drop_stale_blobs()

    def _blob_path(self, T: int) -> str:
        return os.path.join(self._cache_dir, f"encoder_T{T}_{self._device}.blob")

    def _drop_stale_blobs(self) -> None:
        """Remove blobs for bucket sizes / precisions no longer configured."""
        import re
        pat = re.compile(
            rf"^encoder_T\d+_{re.escape(self._device)}(_[a-z0-9]+)?\.blob$"
        )
        valid = {os.path.basename(self._blob_path(T)) for T in self._buckets}
        try:
            for fname in os.listdir(self._cache_dir):
                if pat.match(fname) and fname not in valid:
                    os.remove(os.path.join(self._cache_dir, fname))
                    _LOGGER.info("[encoder] Removed stale blob %s", fname)
        except FileNotFoundError:
            pass

    def _build(self, T: int) -> _SingleEncoder:
        os.makedirs(self._cache_dir, exist_ok=True)
        core = ov.Core()

        # Fast path: a previously exported (or pre-built) blob loads directly,
        # without pulling the multi-GB FP32 ONNX into memory.
        blob_path = self._blob_path(T)
        if os.path.isfile(blob_path) and os.path.getsize(blob_path) > 0:
            _LOGGER.info("[encoder] Importing precompiled blob %s ...", blob_path)
            stream = _MmapBlob(blob_path)
            try:
                compiled = core.import_model(stream, self._device)
            finally:
                stream.close()
            return _SingleEncoder(compiled, T)

        _LOGGER.info("[encoder] Loading ONNX %s ...", self._onnx_path)
        model = core.read_model(self._onnx_path)
        model.reshape({"audio_signal": [1, 128, T], "length": [1]})
        _LOGGER.info("[encoder] Compiling bucket T=%d for %s ...", T, self._device)
        compiled = core.compile_model(
            model, self._device, config=_build_compile_cfg(self._device, self._cache_dir)
        )
        try:
            blob = compiled.export_model()
            # export_model() returns BytesIO on current OpenVINO, bytes on older
            data = blob.getvalue() if hasattr(blob, "getvalue") else blob
            with open(blob_path, "wb") as f:
                f.write(data)
            _LOGGER.info("[encoder] Exported blob to %s for fast warm starts", blob_path)
        except Exception as e:  # noqa: BLE001 — export is an optimization, never fatal
            _LOGGER.warning("[encoder] Blob export failed (%s); warm starts stay slow", e)
        return _SingleEncoder(compiled, T)

    # ORT-compatible interface so onnx_asr can call us as if we were an InferenceSession
    def get_inputs(self):
        return [_Spec(i.any_name, list(i.shape)) for i in self._inputs_meta]

    def get_outputs(self):
        return [_Spec(o.any_name, list(o.shape)) for o in self._outputs_meta]

    def _pick(self, real_T: int) -> _SingleEncoder:
        target_T = next((T for T in self._sorted_Ts if real_T <= T), None)
        if target_T is None:
            target_T = self._sorted_Ts[-1]
            _LOGGER.warning(
                "[encoder] audio %d frames > largest bucket %d — truncating",
                real_T, target_T,
            )
        if self._buckets[target_T] is None:
            _LOGGER.info(
                "[encoder] Lazy-loading bucket T=%d on first request", target_T
            )
            self._buckets[target_T] = self._build(target_T)
        return self._buckets[target_T]

    def run(self, output_names, feed):
        audio = feed["audio_signal"]
        length = feed["length"]
        b, m, real_T = audio.shape
        bucket = self._pick(real_T)
        out = bucket.infer(audio, int(length[0]))
        res = []
        for n in output_names:
            if n in out:
                res.append(out[n])
            elif n == "outputs":
                res.append(next(a for a in out.values() if a.ndim == 3))
            elif n == "encoded_lengths":
                res.append(next(a for a in out.values() if a.ndim == 1))
            else:
                raise KeyError(n)
        return res


class OpenVinoDecoderShim:
    """Decoder/joint shim — static input shapes, called per token in the TDT loop."""

    def __init__(self, ir_path: str, device: str, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        core = ov.Core()
        _LOGGER.info("[decoder] Loading IR %s ...", ir_path)
        model = core.read_model(ir_path)
        _LOGGER.info("[decoder] Compiling for %s ...", device)
        self._compiled = core.compile_model(
            model, device, config=_build_compile_cfg(device, cache_dir)
        )
        self._req = self._compiled.create_infer_request()
        self._out_names = [o.any_name for o in self._compiled.outputs]

    def get_inputs(self):
        return [_Spec(i.any_name, list(i.shape)) for i in self._compiled.inputs]

    def get_outputs(self):
        return [_Spec(o.any_name, list(o.shape)) for o in self._compiled.outputs]

    def run(self, output_names, feed):
        feed_typed = {
            "encoder_outputs": np.asarray(feed["encoder_outputs"], dtype=np.float32),
            "targets": np.asarray(feed["targets"], dtype=np.int32),
            "target_length": np.asarray(feed["target_length"], dtype=np.int32),
            "input_states_1": np.asarray(feed["input_states_1"], dtype=np.float32),
            "input_states_2": np.asarray(feed["input_states_2"], dtype=np.float32),
        }
        result = self._req.infer(feed_typed)
        out = {k.any_name: np.array(v) for k, v in result.items()}
        res = []
        for n in output_names:
            if n in out:
                res.append(out[n])
            elif n == "outputs":
                res.append(next(a for a in out.values() if a.ndim == 4))
            elif n == "output_states_1":
                cands = sorted(
                    [(nm, a) for nm, a in out.items() if a.ndim == 3 and a.shape[0] == 2]
                )
                res.append(cands[0][1])
            elif n == "output_states_2":
                cands = sorted(
                    [(nm, a) for nm, a in out.items() if a.ndim == 3 and a.shape[0] == 2]
                )
                res.append(cands[-1][1])
            else:
                raise KeyError(n)
        return res

# Changelog

## 1.2.0

- Warm starts no longer read the 2.4 GB FP32 encoder: after a successful
  compile the NPU blob is exported to `/data/ov_cache/encoder_T<frames>_<device>.blob`
  and imported directly on later starts (~1.5 GB peak instead of ~3.5 GB).
- Prebuilt blobs listed in `blobs.json` (SHA-256 verified) are downloaded by
  the bootstrap, so listed bucket sizes never need an on-device compile at all.
  Prebuilt blobs are provided for 6, 7, 8, 9 and 10 second buckets
  (compiled offline for NPU 3720 — Meteor/Arrow Lake).
- Stale blobs for no-longer-configured buckets are cleaned up at startup.
- `openvino` pinned to 2026.2.1: blobs are only importable by the exact
  OpenVINO version that exported them.
- Intel NPU driver updated to v1.33.0 — same version the prebuilt blobs are
  compiled with; also ships the NPU compiler loader that OpenVINO 2026.2
  expects for ahead-of-time compilation.

## 1.1.2

- Pull the ubuntu:24.04 base image from mirror.gcr.io (Google's Docker Hub
  library mirror) instead of docker.io, which is unreachable or unstable on
  some networks.

## 1.1.1

- Remove the supervisor watchdog and Docker HEALTHCHECK: the first start
  downloads ~3.2 GB of models before port 10300 opens, so the probes could
  restart the add-on mid-download and caused a harmless but confusing
  "Timeout while waiting for app to start" warning.

## 1.1.0

- Image is now built entirely from source code vendored in this repository
  (`parakeet/`, upstream commit `238b52b`, MIT) — no prebuilt third-party
  images are pulled.
- Intel NPU driver and Level Zero loader downloads are pinned and verified
  by SHA-256 at build time.

## 1.0.0

- Initial release.
- Wraps `ghcr.io/cibernox/wyoming-parakeet-on-intel-npu:latest`.
- Options: `language`, `device`, `encoder_buckets`, `encoder_lazy_buckets`.
- Maps `/dev/accel/accel0` (Intel NPU) into the container.
- Registers Wyoming discovery with Home Assistant automatically.

# Changelog

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

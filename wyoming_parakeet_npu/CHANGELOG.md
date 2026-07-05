# Changelog

## 1.0.0

- Initial release.
- Wraps `ghcr.io/cibernox/wyoming-parakeet-on-intel-npu:latest`.
- Options: `language`, `device`, `encoder_buckets`, `encoder_lazy_buckets`.
- Maps `/dev/accel/accel0` (Intel NPU) into the container.
- Registers Wyoming discovery with Home Assistant automatically.

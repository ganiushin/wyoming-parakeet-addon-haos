# Wyoming Parakeet (Intel NPU) — built entirely from sources in this repository.
# The application code lives in ./parakeet (vendored, MIT — see parakeet/VENDORED.md).
# The only build-time downloads are Intel's official NPU driver packages and the
# oneAPI Level Zero loader, pinned by version and verified by SHA-256.
FROM ubuntu:24.04

ARG LIBZE_URL="https://github.com/oneapi-src/level-zero/releases/download/v1.28.6/libze1_1.28.6+u24.04_amd64.deb"
ARG LIBZE_SHA256="8af9dc06d9684a20a3f43754fe109968fba2ba085cfa572a4cc6ef0272f9bcd1"
ARG NPU_DRIVER_URL="https://github.com/intel/linux-npu-driver/releases/download/v1.32.1/linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz"
ARG NPU_DRIVER_SHA256="c4ad38bad6e1cdc609f68044ea3502366013e6a22758f698ffb532f48bd1ee48"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/data/huggingface \
    TZ=Etc/UTC \
    DATA_DIR=/data \
    LANGUAGE=en \
    DEVICE=NPU \
    ENCODER_BUCKETS=5 \
    ENCODER_LAZY_BUCKETS=20

# System packages (jq is used by the add-on's run.sh to read options.json)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq tzdata \
        python3 python3-pip python3-venv \
        libtbb12 libtbbmalloc2 ocl-icd-libopencl1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Level Zero loader >= 1.17 (Ubuntu 24.04 ships 1.16, which can't discover
# the modern intel-level-zero-npu library)
RUN mkdir /tmp/lz && cd /tmp/lz \
    && curl -fL -o libze1.deb "$LIBZE_URL" \
    && echo "$LIBZE_SHA256  libze1.deb" | sha256sum -c - \
    && apt-get update \
    && apt-get install -y --no-install-recommends ./libze1.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/lz

# Intel NPU userspace driver + firmware + compiler
RUN mkdir /tmp/npu && cd /tmp/npu \
    && curl -fL -o npu.tar.gz "$NPU_DRIVER_URL" \
    && echo "$NPU_DRIVER_SHA256  npu.tar.gz" | sha256sum -c - \
    && tar xzf npu.tar.gz \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ./intel-driver-compiler-npu_*.deb \
        ./intel-fw-npu_*.deb \
        ./intel-level-zero-npu_*.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/npu

# Python environment + the vendored application
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY parakeet/pyproject.toml ./
COPY parakeet/src ./src
COPY parakeet/scripts ./scripts
COPY parakeet/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 10300

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 10300)); s.close()" || exit 1

COPY run.sh /run.sh
RUN chmod +x /run.sh

ENTRYPOINT ["/run.sh"]

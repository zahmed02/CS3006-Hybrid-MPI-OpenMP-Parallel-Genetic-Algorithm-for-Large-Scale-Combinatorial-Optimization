# ============================================================================
# Reproducible build environment for the Parallel GA project
# ============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        git \
        ccache \
        openmpi-bin \
        libopenmpi-dev \
        libomp-dev \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j

CMD ["/workspace/build/bin/pdc_ga", "--cities", "1000", "--generations", "50"]
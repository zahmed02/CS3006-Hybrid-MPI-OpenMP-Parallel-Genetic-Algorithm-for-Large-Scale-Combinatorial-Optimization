# ============================================================================
# Reproducible build + runtime environment for the Parallel GA project.
#
# Build (from repo root):
#   docker build -t pdc-ga -f docker/Dockerfile .
#
# Run (generates a synthetic instance inside the container):
#   docker run --rm pdc-ga
#
# Run with a specific config:
#   docker run --rm pdc-ga /workspace/build/bin/pdc_ga \
#       --generate --cities 2000 --generations 100 --pop 500
# ============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# Allow mpirun to run as root inside the container. Containers always start
# as UID 0, and OpenMPI's root protection is meant for shared HPC clusters,
# not isolated containers. Two ENV vars make any mpirun invocation work.
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ccache \
        openmpi-bin \
        libopenmpi-dev \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only the files needed for the build, then build.
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/     ./src/
COPY tests/   ./tests/

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)"

# Default: run a small self-contained benchmark inside the container
CMD ["mpirun", "--allow-run-as-root", "-np", "2", "--oversubscribe", \
     "/workspace/build/bin/pdc_ga", \
     "--generate", "--cities", "2000", "--generations", "100", "--pop", "500"]
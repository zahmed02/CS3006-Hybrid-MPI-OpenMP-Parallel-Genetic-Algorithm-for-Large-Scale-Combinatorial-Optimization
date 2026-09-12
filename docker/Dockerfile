# Build and run environment for the Parallel GA project.
#
# Build:
#   docker build -t pdc-ga -f docker/Dockerfile .
# Run:
#   docker run --rm pdc-ga
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Let mpirun run as root inside the container.
# Containers start as root, and OpenMPI blocks root by default.
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Install the build toolchain
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

# Copy only the sources needed for the build
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/     ./src/
COPY tests/   ./tests/

# Configure and build
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)"

# Default command: run a small self-contained benchmark
CMD ["mpirun", "--allow-run-as-root", "-np", "2", "--oversubscribe", \
     "/workspace/build/bin/pdc_ga", \
     "--generate", "--cities", "2000", "--generations", "100", "--pop", "500"]
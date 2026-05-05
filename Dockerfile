FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git wget unzip ca-certificates \
    libwxgtk3.0-gtk3-dev libgtk-3-dev \
    libopencv-dev libjpeg-dev libpng-dev libtiff-dev \
    python3 python3-pip \
    libdlib-dev \
    && rm -rf /var/lib/apt/lists/*

# Download libtorch (CPU) into /opt/libtorch
RUN wget -q -O /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

WORKDIR /workspace

# install CMake v3.22+ if needed - using distro cmake here

COPY . /workspace

RUN mkdir -p build && cd build \
    && cmake -DENABLE_LIBTORCH=ON .. \
    && cmake --build . -- -j$(nproc)

CMD ["/workspace/build/hpfrec"]

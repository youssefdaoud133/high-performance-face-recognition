#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
cd build
cmake -DENABLE_LIBTORCH=ON ..
cmake --build . -- -j$(nproc)

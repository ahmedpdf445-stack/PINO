#!/usr/bin/env bash
set -euo pipefail
if [ ! -f pino_kernel.cu ]; then
  echo "ERROR: pino_kernel.cu not found in the current directory." >&2
  exit 1
fi

nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC pino_kernel.cu -lcufft -o libpino_kernel.so

echo "Build complete: libpino_kernel.so"

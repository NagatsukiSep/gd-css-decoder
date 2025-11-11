#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++-14}
NVCC=${NVCC:-nvcc}
EIGEN1="/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
EIGEN2="/usr/local/include/eigen3"
EIGEN3="/usr/include/eigen3"

INC=""
for d in "$EIGEN1" "$EIGEN2" "$EIGEN3"; do
  [[ -d "$d" ]] && INC="$INC -I$d"
done

USE_CUDA_BUILD=${USE_CUDA:-0}

if [[ "$USE_CUDA_BUILD" == "1" ]]; then
  if command -v "$NVCC" >/dev/null 2>&1; then
    mkdir -p build
    echo "ℹ️ Building with CUDA acceleration"
    "$NVCC" -std=c++17 -O3 -DUSE_CUDA -c src/checkpass_cuda.cu -o build/checkpass_cuda.o
    "$CXX" -w -std=c++23 src/gd_css_patched.cc build/checkpass_cuda.o -o gd_css -O3 -lm $INC -D_Alignof=alignof -fopenmp -DUSE_CUDA -lcudart
    echo "✅ Build complete (CUDA): ./gd_css"
    exit 0
  else
    echo "⚠️ USE_CUDA=1 but nvcc was not found. Falling back to CPU build."
  fi
fi

"$CXX" -w -std=c++23 src/gd_css_patched.cc -o gd_css -O3 -lm $INC -D_Alignof=alignof -fopenmp
echo "✅ Build complete: ./gd_css"

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

if command -v "$NVCC" >/dev/null 2>&1; then
  echo "🔧 CUDA toolchain detected: $NVCC"
  BUILD_DIR="${BUILD_DIR:-build}"
  mkdir -p "$BUILD_DIR"

  echo "Compiling CUDA kernels..."
  $NVCC -std=c++17 -c src/gd_css_patched.cu -o "$BUILD_DIR/gd_css_patched_cuda.o" $INC

  echo "Compiling host sources..."
  $CXX -w -std=c++23 -c src/gd_css_patched.cc -o "$BUILD_DIR/gd_css_patched.o" -O3 $INC -D_Alignof=alignof -fopenmp -DUSE_CUDA

  echo "Linking..."
  $NVCC -std=c++17 "$BUILD_DIR/gd_css_patched_cuda.o" "$BUILD_DIR/gd_css_patched.o" -o gd_css -O3 -lm -Xcompiler "-fopenmp"
else
  echo "ℹ️ CUDA toolchain not found. Building CPU-only binary."
  $CXX -w -std=c++23 src/gd_css_patched.cc -o gd_css -O3 -lm $INC -D_Alignof=alignof -fopenmp
fi
echo "✅ Build complete: ./gd_css"

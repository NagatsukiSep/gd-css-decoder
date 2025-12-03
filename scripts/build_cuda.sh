#!/usr/bin/env bash
set -euo pipefail

NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_70}

EIGEN1="/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
EIGEN2="/usr/local/include/eigen3"
EIGEN3="/usr/include/eigen3"
INC=""
for d in "$EIGEN1" "$EIGEN2" "$EIGEN3"; do
  [[ -d "$d" ]] && INC="$INC -I$d"
done

OUTPUT=${1:-gd_css_cuda}

$NVCC -w -x cu -std=c++17 -O3 -Xcompiler "-w" -Xcompiler "-D_Alignof=alignof" \
  -arch=${CUDA_ARCH} -lcudart -lm -DGD_CSS_CUDA_BUILD $INC \
  src/gd_css_patched.cc -o "$OUTPUT"

echo "✅ CUDA build complete: ./$OUTPUT"

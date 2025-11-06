#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++-14}
if ! command -v "$CXX" >/dev/null 2>&1; then
  if command -v g++ >/dev/null 2>&1; then
    CXX=g++
    echo "⚠️ Falling back to g++" >&2
  else
    echo "❌ No suitable C++ compiler found" >&2
    exit 1
  fi
fi
EIGEN1="/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
EIGEN2="/usr/local/include/eigen3"
EIGEN3="/usr/include/eigen3"

INC=""
for d in "$EIGEN1" "$EIGEN2" "$EIGEN3"; do
  [[ -d "$d" ]] && INC="$INC -I$d"
done

$CXX -w -std=c++23 src/gd_css_patched.cc src/check_pass_ems.cc -o gd_css -O3 -lm $INC -D_Alignof=alignof -fopenmp
echo "✅ Build complete: ./gd_css"

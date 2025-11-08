# GD-CSS Decoder (Quantum Error Correction using Non-binary LDPC over GF(q))

This repository provides a C++23 implementation of a **joint / degenerate decoder**
for CSS codes constructed from non-binary LDPC codes over GF(q).
It performs iterative BP decoding with cross-channel coupling (C↔D) and
supports small-error recovery heuristics based on UTCBC structures.

---

## 背景と目的

量子誤り訂正 (Quantum Error Correction; QEC) は，量子ビットが外界との相互作用によって
容易に劣化するという問題を克服し，大規模でフォールトトレラントな量子計算を実現する
ために不可欠な技術です。中でも CSS 型量子 LDPC 符号は，疎な構造を活かした確率伝搬
(Belief Propagation; BP) による反復復号が可能であり，理論・実装の両面で有望視されて
います。

近年，従来の BP 復号を拡張し，縮退性を考慮して X および Z タイプのシンドロームを
同時に処理する *同時 BP 復号* が提案されました。この手法は非二元 LDPC-CSS 符号に
対しハッシング限界近傍の高い性能を示し，従来の独立 BP 復号を大きく上回る復号能力
を有します。しかし，多元有限体 GF(q) 上でのメッセージ伝搬は計算量が非常に大きく，
CPU 実装のみでは大規模符号の高速シミュレーションが困難でした。

本実装では，このボトルネックを解消するために，従来の OpenMP ベース CPU 実装に
加えて GPU 上の並列処理を導入しています。`gd_css_patched.cc` では変数ノード→チェック
ノード更新 (VN→CN) とチェックノード→変数ノード更新 (CheckPass) の双方に CUDA カーネル
を実装し，共有メモリやルックアップテーブルの再利用によって GF(q) 演算を効率化しま
した。さらに，ホスト↔デバイス間のデータ転送 (H2D / D2H) と GPU カーネル実行時間を
分離して計測するデバッグログを備え，性能解析を容易にしています。

本リポジトリを用いることで，非二元量子 LDPC 符号の大規模復号シミュレーションを
GPU 上で実行でき，有限長性能やエラーフロア特性に関する詳細な解析が可能になります。
公開された CUDA 対応実装が高性能量子 LDPC 復号の研究と実装技術の橋渡しになることを
目指しています。

---

## 🧩 Dependencies

This program depends on the following library:

- **Eigen** ≥ 3.4  
  Header-only C++ template library for linear algebra.  
  It is required for matrix and vector operations used in the decoder.

Eigen is **not bundled** with this repository.  
Please install it before building.

### macOS (Homebrew)
```
brew install eigen
```

### Ubuntu / Debian
```
sudo apt update
sudo apt install libeigen3-dev
```

After installation, the build script (`scripts/build.sh`) automatically detects  
the installed Eigen path (`/opt/homebrew/include/eigen3`, `/usr/include/eigen3`, etc.),  
so no manual configuration is required.

---

## ⚙️ Build

### CPU build

To compile the decoder for CPU execution, run:
```
scripts/build.sh
```

Or manually:
```
g++-14 -w -std=c++23 src/gd_css_patched.cc -o gd_css -O3 -lm   -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3   -I/usr/local/include/eigen3   -D_Alignof=alignof
```

### CUDA build

If you have the NVIDIA CUDA toolkit installed and would like to enable the GPU kernels,
use the dedicated build script:
```
scripts/build_cuda.sh
```

By default, the script targets `sm_70`. You can override the GPU architecture and the
output binary name via environment variables and arguments, for example:
```
CUDA_ARCH=sm_80 scripts/build_cuda.sh gd_css_cuda
```

The CUDA build defines the `GD_CSS_CUDA_BUILD` macro and forces nvcc to treat
`src/gd_css_patched.cc` as CUDA code, ensuring the `#if defined(__CUDACC__)` guarded
paths (GPU kernels) are compiled in.

---

## ▶️ Run Example

```
scripts/run_example.sh [binary]
```

If you omit `[binary]`, the script now prefers `./gd_css_cuda` when it exists and falls
back to the CPU build (`./gd_css`). The script also echoes the binary it is about to run,
so you can double-check which build is being executed.

Under the hood, the script executes:

```
./gd_css 500   data/apm_css/DEG_APM_Gamma_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014   data/apm_css/DEG_APM_Delta_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014   DEG_APM_J2_L6_P6500   0.0640   0   101
```

---

## 📊 Output

During decoding, the program prints progress information such as iteration count,
syndrome satisfaction, and error statistics.
Results are also written to log files in the working directory:

- `LOG_*` → decoding statistics (FER/SER, iterations, etc.)
- `EF_LOG_*` → detailed debugging information (error-floor analysis)

When the CUDA implementation of `CheckPass` is active, an additional debug line
summarises GPU timings:

- **H2D transfers** — cumulative time spent copying all inputs from host memory
  to the GPU (CN↔VN message buffers, parity-check structure, GF tables, etc.).
- **kernel** — time measured with CUDA events for the GPU kernel execution.
- **D2H transfers** — cumulative time required to copy the updated message
  buffers back from the GPU to host memory.
- **total transfer** — `H2D + D2H`, highlighting the communication overhead
  relative to the kernel runtime.

---

## 📄 Notes

- Ensure that the following directories exist and contain the required data:
  - `data/apm_css/` — contains matrix files for C and D
  - `data/Tables/` — contains GF(q) lookup tables  
    (`BINGF256`, `ADDGF256`, `MULGF256`, `DIVGF256`, `TENSORFFT256`)
- If the `Tables` directory is inside `data/`, create a symlink for runtime:
  ```
  ln -sfn data/Tables Tables
  ```

---

## 🧠 Citation

If you use this code in academic work, please cite this repository or the related publications.

---

## 📜 License

MIT License © 2025 Kenta Kasai

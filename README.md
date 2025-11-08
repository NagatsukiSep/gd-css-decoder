# GD-CSS Decoder (Quantum Error Correction using Non-binary LDPC over GF(q))

This repository provides a C++23 implementation of a **joint / degenerate decoder**
for CSS codes constructed from non-binary LDPC codes over GF(q).
It performs iterative BP decoding with cross-channel coupling (C↔D) and
supports small-error recovery heuristics based on UTCBC structures.

---

## 背景と目的

量子誤り訂正 (Quantum Error Correction; QEC) は，量子ビットが外界との相互作用に起因する
劣化を抑制し，フォールトトレラントな大規模量子計算を実現するための基盤技術である。
特に Calderbank–Shor–Steane (CSS) 型量子 LDPC 符号は，疎行列を活用した確率伝搬
(Belief Propagation; BP) に基づく反復復号が可能であり，符号設計の柔軟性と実装効率の
両立を図れる点で注目を集めている。

従来の CSS 復号では X および Z タイプの誤り推定を別々に扱う枠組みが主流であったが，
縮退性を陽に考慮して両シンドロームを同時に処理する *同時 BP 復号* が提案され，
非二元 LDPC-CSS 符号に対してハッシング限界近傍の性能を達成することが報告されている。
一方で，多元有限体 GF(q) 上のメッセージ更新は演算量・メモリアクセス量の両面で
膨大であり，CPU のみの実装では大規模符号の性能評価やパラメータ探索を高速に行うことが
難しい。

本研究では，上記課題に対応するため，OpenMP による従来の CPU 実装を土台としつつ，
CUDA を用いた GPU 並列化を導入した。同時 BP 復号における変数ノード→チェックノード
(VN→CN) 更新およびチェックノード→変数ノード (CheckPass) 更新に対して専用 CUDA カーネルを
設計し，共有メモリを活用した GF(q) テーブル参照の最適化やメッセージバッファの再利用に
よって並列効率を高めている。さらに，ホスト↔デバイス間データ転送 (H2D / D2H) と
カーネル実行を個別に計測する計時機構を実装し，GPU アクセラレーションの効果検証と
ボトルネック分析を可能とした。

本リポジトリは，非二元量子 LDPC 符号に対する同時 BP 復号を GPU 上で実現するための
リファレンス実装として提供される。研究者・実務者は本実装を活用することで，大規模な
復号シミュレーションを効率的に実施し，有限長性能やエラーフロア特性に関するより詳細な
解析を行えると期待される。また，ソースコードを公開することで，高性能量子 LDPC 復号の
研究開発を促進し，理論研究とシステム実装との橋渡しに資することを目的とする。

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

# GD-CSS Decoder (Quantum Error Correction using Non-binary LDPC over GF(q))

This repository provides a C++23 implementation of a **joint / degenerate decoder**  
for CSS codes constructed from non-binary LDPC codes over GF(q).  
It performs iterative BP decoding with cross-channel coupling (C↔D) and  
supports small-error recovery heuristics based on UTCBC structures.

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
scripts/run_example.sh [IS_PRINTED_TIME] [DECODE_COUNT]
```

If you omit `[binary]`, the script now prefers `./gd_css_cuda` when it exists and falls
back to the CPU build (`./gd_css`). The script also echoes the binary it is about to run,
so you can double-check which build is being executed.

Under the hood, the script executes:

```
./gd_css 500   data/apm_css/DEG_APM_Gamma_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014   data/apm_css/DEG_APM_Delta_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014   DEG_APM_J2_L6_P6500   0.0640   0   101   1   10
```

---

## 📊 Output

During decoding, the program prints progress information such as iteration count,
syndrome satisfaction, and error statistics.
Results are also written to log files in the working directory:

- `LOG_*` → decoding statistics (FER/SER, iterations, etc.)
- `EF_LOG_*` → detailed debugging information (error-floor analysis)

### Measurement mode (latency & throughput)

You can measure per-decode latency and logical qubit throughput with the optional
`measurement_runs` argument:

```
./gd_css max_iter filename_C filename_D logfile f_m DEBUG_transmission seed [timing_debug] [measurement_runs]
```

- `timing_debug` (optional): set to `1` to enable verbose timing prints.
- `measurement_runs` (optional):
  - `> 0` — measure exactly this many decode runs, then stop.
  - `0` — disable measurement (legacy behavior).
  - `< 0` or omitted — measure indefinitely until you manually stop the process.

The measurement summary reports total/average runtime per decode and the
logical-qubits-per-second throughput computed as:

```
logical_qubits_per_decode = logGF * P * L * (M / N)
```

where `P`, `L`, `M`, `N`, and `logGF` are derived from the input matrices.

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

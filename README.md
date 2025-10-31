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

---

## ▶️ Run Example

```
scripts/run_example.sh
```

Equivalent to:
```
./gd_css 500   data/apm_css/QC_Gamma_J2_L6_P6500_RQ0.333333_alpha1_GF256_GIRTH12_SEED101   data/apm_css/QC_Delta_J2_L6_P6500_RQ0.333333_alpha1_GF256_GIRTH12_SEED101   QC_J2_L6_P6500   0.0640   1   101
```

---

## 📊 Output

During decoding, the program prints progress information such as iteration count,  
syndrome satisfaction, and error statistics.  
Results are also written to log files in the working directory:

- `LOG_*` → decoding statistics (FER/SER, iterations, etc.)  
- `EF_LOG_*` → detailed debugging information (error-floor analysis)

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

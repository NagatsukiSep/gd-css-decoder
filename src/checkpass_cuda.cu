#include "checkpass_cuda.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline int roundup_pow2(int v) {
    int power = 1;
    while (power < v) {
        power <<= 1;
    }
    return power;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& message)
        : std::runtime_error(message) {}
};

inline void CheckCuda(cudaError_t result, const char* expr) {
    if (result != cudaSuccess) {
        throw CudaError(std::string("CUDA error: ") + cudaGetErrorString(result) + " in " + expr);
    }
}

struct DeviceBuffer {
    void* ptr{nullptr};
    size_t bytes{0};

    DeviceBuffer() = default;
    DeviceBuffer(size_t size, size_t alignment = alignof(double)) {
        allocate(size, alignment);
    }

    void allocate(size_t size, size_t /*alignment*/ = alignof(double)) {
        release();
        if (size == 0) {
            return;
        }
        bytes = size;
        CheckCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    }

    void release() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }

    ~DeviceBuffer() { release(); }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }

    template <typename T>
    const T* as() const {
        return static_cast<const T*>(ptr);
    }
};

struct ScopedDeviceSetter {
    int original{-1};
    explicit ScopedDeviceSetter(int device = 0) {
        CheckCuda(cudaGetDevice(&original), "cudaGetDevice");
        if (device != original) {
            CheckCuda(cudaSetDevice(device), "cudaSetDevice");
        }
    }
    ~ScopedDeviceSetter() {
        if (original >= 0) {
            cudaSetDevice(original);
        }
    }
};

__global__ void CheckPassKernel(
    double* d_CNtoVN,
    double* d_VNtoCN,
    const int* d_matValue,
    const int* d_rowBase,
    const int* d_rowDegree,
    const int* d_mulGF,
    const int* d_divGF,
    const int2* d_fftPairs,
    const int* d_trueNoise,
    int M,
    int GF,
    int logGF) {

    int m = blockIdx.x;
    if (m >= M) {
        return;
    }

    int tid = threadIdx.x;
    int deg = d_rowDegree[m];
    if (deg == 0) {
        return;
    }
    int base = d_rowBase[m];
    int pairsPerStage = GF / 2;

    extern __shared__ double shared[];
    double* F = shared;                       // GF entries
    double* tmp = F + GF;                     // GF entries
    double* workspace = tmp + GF;             // 2 * pairsPerStage entries
    double* sumBuf = workspace + (pairsPerStage * 2); // 1 entry

    for (int g = tid; g < GF; g += blockDim.x) {
        F[g] = (g == d_trueNoise[m]) ? 1.0 : 0.0;
    }
    __syncthreads();

    for (int stage = 0; stage < logGF; ++stage) {
        int offset = stage * pairsPerStage;
        for (int idx = tid; idx < pairsPerStage; idx += blockDim.x) {
            int2 pr = d_fftPairs[offset + idx];
            double A = F[pr.x];
            double B = F[pr.y];
            workspace[2 * idx] = A + B;
            workspace[2 * idx + 1] = A - B;
        }
        __syncthreads();
        for (int idx = tid; idx < pairsPerStage; idx += blockDim.x) {
            int2 pr = d_fftPairs[offset + idx];
            F[pr.x] = workspace[2 * idx];
            F[pr.y] = workspace[2 * idx + 1];
        }
        __syncthreads();
    }

    for (int t = 0; t < deg; ++t) {
        int edgeIdx = base + t;
        double* vn = d_VNtoCN + static_cast<size_t>(edgeIdx) * GF;
        int coeff = d_matValue[edgeIdx];

        for (int g = tid; g < GF; g += blockDim.x) {
            int idx = d_divGF[g * GF + coeff];
            tmp[g] = vn[idx];
        }
        __syncthreads();
        for (int g = tid; g < GF; g += blockDim.x) {
            vn[g] = tmp[g];
        }
        __syncthreads();

        for (int stage = 0; stage < logGF; ++stage) {
            int offset = stage * pairsPerStage;
            for (int idx = tid; idx < pairsPerStage; idx += blockDim.x) {
                int2 pr = d_fftPairs[offset + idx];
                double A = vn[pr.x];
                double B = vn[pr.y];
                workspace[2 * idx] = A + B;
                workspace[2 * idx + 1] = A - B;
            }
            __syncthreads();
            for (int idx = tid; idx < pairsPerStage; idx += blockDim.x) {
                int2 pr = d_fftPairs[offset + idx];
                vn[pr.x] = workspace[2 * idx];
                vn[pr.y] = workspace[2 * idx + 1];
            }
            __syncthreads();
        }
    }

    __syncthreads();

    for (int t = 0; t < deg; ++t) {
        int edgeIdx = base + t;
        double* cn = d_CNtoVN + static_cast<size_t>(edgeIdx) * GF;

        for (int g = tid; g < GF; g += blockDim.x) {
            cn[g] = F[g];
        }
        __syncthreads();

        for (int tz = 0; tz < deg; ++tz) {
            if (tz == t) {
                continue;
            }
            int otherEdge = base + tz;
            double* other = d_VNtoCN + static_cast<size_t>(otherEdge) * GF;
            for (int g = tid; g < GF; g += blockDim.x) {
                cn[g] *= other[g];
            }
            __syncthreads();
        }
    }

    __syncthreads();

    for (int t = 0; t < deg; ++t) {
        int edgeIdx = base + t;
        double* cn = d_CNtoVN + static_cast<size_t>(edgeIdx) * GF;
        int coeff = d_matValue[edgeIdx];

        for (int stage = 0; stage < logGF; ++stage) {
            int offset = stage * pairsPerStage;
            for (int idx = tid; idx < pairsPerStage; idx += blockDim.x) {
                int2 pr = d_fftPairs[offset + idx];
                double A = cn[pr.x];
                double B = cn[pr.y];
                workspace[2 * idx] = 0.5 * (A + B);
                workspace[2 * idx + 1] = 0.5 * (A - B);
            }
            __syncthreads();
            for (int idx = tid; idx < pairsPerStage; idx += blockDim.x) {
                int2 pr = d_fftPairs[offset + idx];
                cn[pr.x] = workspace[2 * idx];
                cn[pr.y] = workspace[2 * idx + 1];
            }
            __syncthreads();
        }

        for (int g = tid; g < GF; g += blockDim.x) {
            int idx = d_mulGF[coeff * GF + g];
            tmp[g] = cn[idx];
        }
        __syncthreads();
        for (int g = tid; g < GF; g += blockDim.x) {
            double val = tmp[g];
            cn[g] = val > 0.0 ? val : 0.0;
        }
        __syncthreads();

        if (tid == 0) {
            sumBuf[0] = 0.0;
        }
        __syncthreads();

        double localSum = 0.0;
        for (int g = tid; g < GF; g += blockDim.x) {
            localSum += cn[g];
        }
        atomicAdd(sumBuf, localSum);
        __syncthreads();

        double total = sumBuf[0];
        if (total == 0.0) {
            double uniform = 1.0 / static_cast<double>(GF);
            for (int g = tid; g < GF; g += blockDim.x) {
                cn[g] = uniform;
            }
        } else {
            double inv = 1.0 / total;
            for (int g = tid; g < GF; g += blockDim.x) {
                cn[g] *= inv;
            }
        }
        __syncthreads();
    }
}

} // namespace

bool CheckPassCUDA(
    std::vector<std::vector<double>>& CNtoVNxxx,
    std::vector<std::vector<double>>& VNtoCNxxx,
    const std::vector<std::vector<int>>& MatValue,
    int M,
    const std::vector<int>& RowDegree,
    const std::vector<std::vector<int>>& MULGF,
    const std::vector<std::vector<int>>& DIVGF,
    const std::vector<std::vector<int>>& FFTSQ,
    int GF,
    const std::vector<int>& TrueNoiseSynd) {

    if (M == 0) {
        return true;
    }

    ScopedDeviceSetter deviceGuard;

    int logGF = static_cast<int>(std::round(std::log2(static_cast<double>(GF))));
    int pairsPerStage = GF / 2;
    size_t totalPairs = static_cast<size_t>(logGF) * static_cast<size_t>(pairsPerStage);

    std::vector<int> rowBase(M + 1, 0);
    for (int m = 1; m <= M; ++m) {
        rowBase[m] = rowBase[m - 1] + RowDegree[m - 1];
    }
    int totalEdges = rowBase[M];

    if (static_cast<int>(CNtoVNxxx.size()) != totalEdges ||
        static_cast<int>(VNtoCNxxx.size()) != totalEdges) {
        throw CudaError("Edge buffer size mismatch with RowDegree totals");
    }

    std::vector<int> matValueFlat(totalEdges);
    for (int m = 0; m < M; ++m) {
        int base = rowBase[m];
        for (int t = 0; t < RowDegree[m]; ++t) {
            matValueFlat[base + t] = MatValue[m][t];
        }
    }

    std::vector<int2> fftPairs(totalPairs);
    for (size_t k = 0; k < totalPairs; ++k) {
        fftPairs[k] = make_int2(FFTSQ[k][0], FFTSQ[k][1]);
    }

    std::vector<int> mulFlat(GF * GF);
    std::vector<int> divFlat(GF * GF);
    for (int i = 0; i < GF; ++i) {
        for (int j = 0; j < GF; ++j) {
            mulFlat[i * GF + j] = MULGF[i][j];
            divFlat[i * GF + j] = DIVGF[i][j];
        }
    }

    std::vector<double> cnFlat(static_cast<size_t>(totalEdges) * GF);
    std::vector<double> vnFlat(static_cast<size_t>(totalEdges) * GF);
    for (int edge = 0; edge < totalEdges; ++edge) {
        std::copy(CNtoVNxxx[edge].begin(), CNtoVNxxx[edge].end(), cnFlat.begin() + static_cast<size_t>(edge) * GF);
        std::copy(VNtoCNxxx[edge].begin(), VNtoCNxxx[edge].end(), vnFlat.begin() + static_cast<size_t>(edge) * GF);
    }

    DeviceBuffer d_CNtoVN(cnFlat.size() * sizeof(double));
    DeviceBuffer d_VNtoCN(vnFlat.size() * sizeof(double));
    DeviceBuffer d_matValue(matValueFlat.size() * sizeof(int));
    DeviceBuffer d_rowBase(rowBase.size() * sizeof(int));
    DeviceBuffer d_rowDegree(RowDegree.size() * sizeof(int));
    DeviceBuffer d_mulGF(mulFlat.size() * sizeof(int));
    DeviceBuffer d_divGF(divFlat.size() * sizeof(int));
    DeviceBuffer d_fftPairs(fftPairs.size() * sizeof(int2));
    DeviceBuffer d_trueNoise(TrueNoiseSynd.size() * sizeof(int));

    CheckCuda(cudaMemcpy(d_CNtoVN.ptr, cnFlat.data(), cnFlat.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_VNtoCN.ptr, vnFlat.data(), vnFlat.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_matValue.ptr, matValueFlat.data(), matValueFlat.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_rowBase.ptr, rowBase.data(), rowBase.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_rowDegree.ptr, RowDegree.data(), RowDegree.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_mulGF.ptr, mulFlat.data(), mulFlat.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_divGF.ptr, divFlat.data(), divFlat.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_fftPairs.ptr, fftPairs.data(), fftPairs.size() * sizeof(int2), cudaMemcpyHostToDevice), "cudaMemcpy");
    CheckCuda(cudaMemcpy(d_trueNoise.ptr, TrueNoiseSynd.data(), TrueNoiseSynd.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");

    int threads = std::min(1024, roundup_pow2(GF));
    size_t sharedBytes = static_cast<size_t>(2 * GF + 2 * pairsPerStage + 1) * sizeof(double);
    CheckPassKernel<<<M, threads, sharedBytes>>>(
        d_CNtoVN.as<double>(),
        d_VNtoCN.as<double>(),
        d_matValue.as<int>(),
        d_rowBase.as<int>(),
        d_rowDegree.as<int>(),
        d_mulGF.as<int>(),
        d_divGF.as<int>(),
        d_fftPairs.as<int2>(),
        d_trueNoise.as<int>(),
        M,
        GF,
        logGF);
    CheckCuda(cudaGetLastError(), "CheckPassKernel");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    CheckCuda(cudaMemcpy(cnFlat.data(), d_CNtoVN.ptr, cnFlat.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    CheckCuda(cudaMemcpy(vnFlat.data(), d_VNtoCN.ptr, vnFlat.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");

    for (int edge = 0; edge < totalEdges; ++edge) {
        std::copy(cnFlat.begin() + static_cast<size_t>(edge) * GF,
                  cnFlat.begin() + static_cast<size_t>(edge + 1) * GF,
                  CNtoVNxxx[edge].begin());
        std::copy(vnFlat.begin() + static_cast<size_t>(edge) * GF,
                  vnFlat.begin() + static_cast<size_t>(edge + 1) * GF,
                  VNtoCNxxx[edge].begin());
    }

    return true;
}

#endif

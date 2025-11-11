#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { reset(); }

  bool allocate(size_t count) {
    reset();
    size_ = count;
    if (count == 0) return true;
    cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
    if (err != cudaSuccess) {
      ptr_ = nullptr;
      size_ = 0;
      return false;
    }
    return true;
  }

  void reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

bool checkCuda(cudaError_t err, const char* label) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA] " << label << ": " << cudaGetErrorString(err) << "\n";
    return false;
  }
  return true;
}

__global__ void ComputeAPPKernel(const double* __restrict__ cn_to_vn,
                                 const double* __restrict__ vn_to_ch,
                                 const int* __restrict__ interleaver,
                                 const int* __restrict__ col_offsets,
                                 const int* __restrict__ col_deg,
                                 double* __restrict__ ch_to_vn,
                                 double* __restrict__ app,
                                 int N,
                                 int GF) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * GF;
  if (idx >= total) return;

  int n = idx / GF;
  int g = idx % GF;
  int start = col_offsets[n];
  int degree = col_deg[n];
  double prod = 1.0;
  for (int t = 0; t < degree; ++t) {
    int edge = interleaver[start + t];
    prod *= cn_to_vn[static_cast<size_t>(edge) * GF + g];
  }
  double ch_val = prod;
  double app_val = ch_val * vn_to_ch[idx];
  ch_to_vn[idx] = ch_val;
  app[idx] = app_val;
}

__global__ void DecisionKernel(const double* __restrict__ app,
                               const int* __restrict__ decision_in,
                               int* __restrict__ decision_out,
                               unsigned char* __restrict__ changed,
                               int N,
                               int GF) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  double max_val = app[n * GF];
  int max_idx = 0;
  for (int g = 1; g < GF; ++g) {
    double val = app[n * GF + g];
    if (val >= max_val) {
      max_val = val;
      max_idx = g;
    }
  }

  int prev = decision_in[n];
  decision_out[n] = max_idx;
  changed[n] = (prev == max_idx) ? 0 : 1;
}

}  // namespace

extern "C" bool InitializeCudaSupport() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    std::cerr << "[CUDA] cudaGetDeviceCount failed: "
              << cudaGetErrorString(err) << "\n";
    return false;
  }
  if (deviceCount <= 0) {
    std::cerr << "[CUDA] No CUDA-capable devices found.\n";
    return false;
  }
  // Trigger lazy initialization.
  cudaFree(nullptr);
  return true;
}

extern "C" bool ComputeAPP_GPU(std::vector<std::vector<double>> &APP,
                                std::vector<std::vector<double>> &ChNtoVN,
                                const std::vector<std::vector<double>> &CNtoVNxxx,
                                const std::vector<std::vector<double>> &VNtoChN,
                                const std::vector<int> &Interleaver,
                                const std::vector<int> &ColDeg,
                                int N,
                                int GF) {
  if (N <= 0 || GF <= 0) return true;
  if (static_cast<int>(ColDeg.size()) < N) return false;
  if (static_cast<int>(VNtoChN.size()) < N) return false;

  size_t totalEdges = Interleaver.size();
  if (CNtoVNxxx.size() < totalEdges) return false;

  std::vector<int> colOffsets(N + 1, 0);
  for (int n = 0; n < N; ++n) {
    colOffsets[n + 1] = colOffsets[n] + ColDeg[n];
  }
  if (colOffsets.back() != static_cast<int>(totalEdges)) {
    return false;
  }

  std::vector<double> flatCN(totalEdges * static_cast<size_t>(GF));
  for (size_t edge = 0; edge < totalEdges; ++edge) {
    if (CNtoVNxxx[edge].size() != static_cast<size_t>(GF)) return false;
    std::copy(CNtoVNxxx[edge].begin(), CNtoVNxxx[edge].end(),
              flatCN.begin() + static_cast<long long>(edge) * GF);
  }

  std::vector<double> flatVN(static_cast<size_t>(N) * GF);
  for (int n = 0; n < N; ++n) {
    if (VNtoChN[n].size() != static_cast<size_t>(GF)) return false;
    std::copy(VNtoChN[n].begin(), VNtoChN[n].end(),
              flatVN.begin() + static_cast<long long>(n) * GF);
  }

  DeviceBuffer<double> d_CN;
  DeviceBuffer<double> d_VN;
  DeviceBuffer<double> d_APP;
  DeviceBuffer<double> d_Ch;
  DeviceBuffer<int> d_interleaver;
  DeviceBuffer<int> d_colOffsets;
  DeviceBuffer<int> d_colDeg;

  if (!d_CN.allocate(flatCN.size()) || !d_VN.allocate(flatVN.size()) ||
      !d_APP.allocate(flatVN.size()) || !d_Ch.allocate(flatVN.size()) ||
      !d_interleaver.allocate(Interleaver.size()) ||
      !d_colOffsets.allocate(colOffsets.size()) ||
      !d_colDeg.allocate(ColDeg.size())) {
    std::cerr << "[CUDA] Memory allocation failed in ComputeAPP_GPU.\n";
    return false;
  }

  if (!checkCuda(cudaMemcpy(d_CN.get(), flatCN.data(),
                            flatCN.size() * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy CNtoVN")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(d_VN.get(), flatVN.data(),
                            flatVN.size() * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy VNtoChN")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(d_interleaver.get(), Interleaver.data(),
                            Interleaver.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy Interleaver")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(d_colOffsets.get(), colOffsets.data(),
                            colOffsets.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy colOffsets")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(d_colDeg.get(), ColDeg.data(),
                            ColDeg.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy ColDeg")) {
    return false;
  }

  int total = N * GF;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ComputeAPPKernel<<<blocks, threads>>>(d_CN.get(), d_VN.get(), d_interleaver.get(),
                                        d_colOffsets.get(), d_colDeg.get(),
                                        d_Ch.get(), d_APP.get(), N, GF);
  if (!checkCuda(cudaGetLastError(), "ComputeAPPKernel launch")) {
    return false;
  }
  if (!checkCuda(cudaDeviceSynchronize(), "ComputeAPPKernel sync")) {
    return false;
  }

  std::vector<double> flatAPP(flatVN.size());
  std::vector<double> flatCh(flatVN.size());
  if (!checkCuda(cudaMemcpy(flatAPP.data(), d_APP.get(),
                            flatAPP.size() * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy APP back")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(flatCh.data(), d_Ch.get(),
                            flatCh.size() * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy ChNtoVN back")) {
    return false;
  }

  if (static_cast<int>(APP.size()) < N) APP.resize(N);
  if (static_cast<int>(ChNtoVN.size()) < N) ChNtoVN.resize(N);

  for (int n = 0; n < N; ++n) {
    APP[n].assign(GF, 0.0);
    ChNtoVN[n].assign(GF, 0.0);
    for (int g = 0; g < GF; ++g) {
      APP[n][g] = flatAPP[static_cast<size_t>(n) * GF + g];
      ChNtoVN[n][g] = flatCh[static_cast<size_t>(n) * GF + g];
    }
  }

  return true;
}

extern "C" bool Decision_GPU(std::vector<int> &Decision,
                              std::vector<int> &Updated_EstmNoise_History,
                              const std::vector<std::vector<double>> &APP,
                              int N,
                              int GF) {
  if (N <= 0 || GF <= 0) {
    Updated_EstmNoise_History.clear();
    return true;
  }
  if (static_cast<int>(APP.size()) < N) return false;
  if (static_cast<int>(Decision.size()) < N) Decision.resize(N, 0);

  std::vector<double> flatAPP(static_cast<size_t>(N) * GF);
  for (int n = 0; n < N; ++n) {
    if (APP[n].size() != static_cast<size_t>(GF)) return false;
    std::copy(APP[n].begin(), APP[n].end(),
              flatAPP.begin() + static_cast<long long>(n) * GF);
  }

  std::vector<int> decisionInput = Decision;
  decisionInput.resize(N, 0);

  DeviceBuffer<double> d_APP;
  DeviceBuffer<int> d_decisionIn;
  DeviceBuffer<int> d_decisionOut;
  DeviceBuffer<unsigned char> d_changed;

  if (!d_APP.allocate(flatAPP.size()) || !d_decisionIn.allocate(N) ||
      !d_decisionOut.allocate(N) || !d_changed.allocate(N)) {
    std::cerr << "[CUDA] Memory allocation failed in Decision_GPU.\n";
    return false;
  }

  if (!checkCuda(cudaMemcpy(d_APP.get(), flatAPP.data(),
                            flatAPP.size() * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy APP")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(d_decisionIn.get(), decisionInput.data(),
                            N * sizeof(int), cudaMemcpyHostToDevice),
                 "cudaMemcpy DecisionIn")) {
    return false;
  }

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  DecisionKernel<<<blocks, threads>>>(d_APP.get(), d_decisionIn.get(),
                                      d_decisionOut.get(), d_changed.get(),
                                      N, GF);
  if (!checkCuda(cudaGetLastError(), "DecisionKernel launch")) {
    return false;
  }
  if (!checkCuda(cudaDeviceSynchronize(), "DecisionKernel sync")) {
    return false;
  }

  std::vector<int> decisionOut(N, 0);
  std::vector<unsigned char> changed(N, 0);
  if (!checkCuda(cudaMemcpy(decisionOut.data(), d_decisionOut.get(),
                            N * sizeof(int), cudaMemcpyDeviceToHost),
                 "cudaMemcpy DecisionOut")) {
    return false;
  }
  if (!checkCuda(cudaMemcpy(changed.data(), d_changed.get(),
                            N * sizeof(unsigned char), cudaMemcpyDeviceToHost),
                 "cudaMemcpy Changed")) {
    return false;
  }

  Decision.assign(decisionOut.begin(), decisionOut.end());
  Updated_EstmNoise_History.clear();
  for (int n = 0; n < N; ++n) {
    if (changed[n]) {
      Updated_EstmNoise_History.push_back(n);
    }
  }
  return true;
}

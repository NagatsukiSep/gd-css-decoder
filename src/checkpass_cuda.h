#pragma once

#include <vector>

#ifdef USE_CUDA
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
    const std::vector<int>& TrueNoiseSynd);
#endif

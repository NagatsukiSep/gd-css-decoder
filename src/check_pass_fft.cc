#include "check_pass_fft.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

void normalize_probabilities(std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        const double uniform = 1.0 / static_cast<double>(values.size());
        std::fill(values.begin(), values.end(), uniform);
        return;
    }
    const double inv = 1.0 / sum;
    for (double& v : values) {
        if (!std::isfinite(v) || v < 0.0) {
            v = 0.0;
        }
        v *= inv;
    }
}

}  // namespace

void CheckPass_FFT(
    std::vector<std::vector<double>>& CNtoVN,
    std::vector<std::vector<double>>& VNtoCN,
    const std::vector<std::vector<int>>& MatValue,
    int M,
    const std::vector<int>& RowDegree,
    const std::vector<std::vector<int>>& MULGF,
    const std::vector<std::vector<int>>& DIVGF,
    const std::vector<std::vector<int>>& FFTSQ,
    int GF,
    const std::vector<int>& TrueNoiseSynd) {

    const int logGF = static_cast<int>(std::round(std::log2(GF)));

    std::vector<int> rowBase(M + 1, 0);
    for (int m = 1; m <= M; ++m) {
        rowBase[m] = rowBase[m - 1] + RowDegree[m - 1];
    }

    std::vector<double> temp(GF, 0.0);
    std::vector<double> syndrome_freq(GF, 0.0);

    for (int m = 0; m < M; ++m) {
        const int base = rowBase[m];
        const int degree = RowDegree[m];
        if (degree <= 0) continue;

        std::fill(syndrome_freq.begin(), syndrome_freq.end(), 0.0);
        syndrome_freq[TrueNoiseSynd[m]] = 1.0;

        for (int k = 0; k < logGF * GF / 2; ++k) {
            const int i = FFTSQ[k][0];
            const int j = FFTSQ[k][1];
            const double A = syndrome_freq[i];
            const double B = syndrome_freq[j];
            syndrome_freq[i] = A + B;
            syndrome_freq[j] = A - B;
        }

        for (int t = 0; t < degree; ++t) {
            const int a = MatValue[m][t];
            for (int g = 0; g < GF; ++g) {
                temp[g] = VNtoCN[base + t][DIVGF[g][a]];
            }
            for (int g = 0; g < GF; ++g) {
                VNtoCN[base + t][g] = temp[g];
            }

            for (int k = 0; k < logGF * GF / 2; ++k) {
                const int i = FFTSQ[k][0];
                const int j = FFTSQ[k][1];
                const double A = VNtoCN[base + t][i];
                const double B = VNtoCN[base + t][j];
                VNtoCN[base + t][i] = A + B;
                VNtoCN[base + t][j] = A - B;
            }
        }

        for (int t = 0; t < degree; ++t) {
            for (int g = 0; g < GF; ++g) {
                CNtoVN[base + t][g] = syndrome_freq[g];
            }

            for (int tz = 0; tz < degree; ++tz) {
                if (tz == t) continue;
                for (int g = 0; g < GF; ++g) {
                    CNtoVN[base + t][g] *= VNtoCN[base + tz][g];
                }
            }
        }

        for (int t = 0; t < degree; ++t) {
            for (int k = 0; k < logGF * GF / 2; ++k) {
                const int i = FFTSQ[k][0];
                const int j = FFTSQ[k][1];
                const double A = CNtoVN[base + t][i];
                const double B = CNtoVN[base + t][j];
                CNtoVN[base + t][i] = 0.5 * (A + B);
                CNtoVN[base + t][j] = 0.5 * (A - B);
            }

            const int a = MatValue[m][t];
            for (int g = 0; g < GF; ++g) {
                temp[g] = CNtoVN[base + t][MULGF[a][g]];
            }
            for (int g = 0; g < GF; ++g) {
                CNtoVN[base + t][g] = std::max(temp[g], 0.0);
            }

            normalize_probabilities(CNtoVN[base + t]);
        }
    }
}


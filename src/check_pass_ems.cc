#include "check_pass_ems.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

extern std::vector<std::vector<int>> ADDGF;

namespace {

static inline double safe_log(double x) {
    const double eps = 1e-300;
    return std::log(std::max(x, eps));
}

static double neglog_sum_exp(double a, double b) {
    if (!std::isfinite(a)) return b;
    if (!std::isfinite(b)) return a;
    double x = -a;
    double y = -b;
    double m = std::max(x, y);
    double sum = std::exp(x - m) + std::exp(y - m);
    return -(m + std::log(sum));
}

static std::vector<std::pair<int, double>> ems_convolve_topk(
    const std::vector<std::pair<int, double>>& A,
    const std::vector<std::pair<int, double>>& B,
    int K) {
    const int aN = static_cast<int>(std::min(A.size(), static_cast<std::size_t>(K)));
    const int bN = static_cast<int>(std::min(B.size(), static_cast<std::size_t>(K)));

    std::unordered_map<int, double> accum;
    accum.reserve(static_cast<std::size_t>(aN * bN));
    for (int i = 0; i < aN; ++i) {
        for (int j = 0; j < bN; ++j) {
            int sym = ADDGF[A[i].first][B[j].first];
            double cost = A[i].second + B[j].second;
            auto [it, inserted] = accum.emplace(sym, cost);
            if (!inserted) {
                it->second = neglog_sum_exp(it->second, cost);
            }
        }
    }

    std::vector<std::pair<int, double>> merged;
    merged.reserve(accum.size());
    for (auto& kv : accum) {
        merged.emplace_back(kv.first, kv.second);
    }

    std::sort(merged.begin(), merged.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    if (static_cast<int>(merged.size()) > K) {
        merged.resize(K);
    }
    return merged;
}

static void list_to_dense(const std::vector<std::pair<int, double>>& L,
                          std::vector<double>& dense) {
    constexpr double INF = std::numeric_limits<double>::infinity();
    std::fill(dense.begin(), dense.end(), INF);
    for (const auto& p : L) {
        dense[p.first] = std::min(dense[p.first], p.second);
    }
}

static void normalize_probabilities(std::vector<double>& input) {
    double sum = 0.0;
    for (double v : input) {
        sum += v;
    }
    if (sum == 0.0) {
        std::cout << "divided by zero" << std::endl;
        double uniform = 1.0 / static_cast<double>(input.size());
        for (double& v : input) {
            v = uniform;
        }
        return;
    }
    for (double& v : input) {
        v /= sum;
    }
}

}  // namespace

void CheckPass_EMS(
  std::vector<std::vector<double>> &CNtoVNxxx,
  std::vector<std::vector<double>> &VNtoCNxxx,
  std::vector<std::vector<int>>    &MatValue,
  int M,
  std::vector<int>            &RowDegree,
  std::vector<std::vector<int>>    &MULGF,
  std::vector<std::vector<int>>    &DIVGF,
  int GF,
  std::vector<int>            &TrueNoiseSynd
){
    const int EMS_K = std::min(24, GF);
    const int K = std::max(1, EMS_K);
    (void)MULGF;

    std::vector<int> rowBase(M + 1, 0);
    for (int m = 1; m <= M; ++m) {
        rowBase[m] = rowBase[m - 1] + RowDegree[m - 1];
    }

    std::vector<double> dense(GF);
    std::vector<double> cost_out_z(GF);
    std::vector<double> dest(GF);

    for (int m = 0; m < M; ++m) {
        const int d = RowDegree[m];
        if (d <= 0) continue;

        const int base = rowBase[m];
        const int synd = TrueNoiseSynd[m];

        std::vector<std::vector<std::pair<int,double>>> top_lists(d);
        std::vector<double> cost(GF);

        for (int t = 0; t < d; ++t) {
            const int a = MatValue[m][t];
            for (int g = 0; g < GF; ++g) {
                double p = VNtoCNxxx[base + t][DIVGF[g][a]];
                cost[g] = -safe_log(p);
            }

            std::vector<int> order(GF);
            std::iota(order.begin(), order.end(), 0);
            const int take = std::min(K, GF);
            std::partial_sort(order.begin(), order.begin() + take, order.end(),
                              [&](int lhs, int rhs){ return cost[lhs] < cost[rhs]; });
            order.resize(take);

            auto &bucket = top_lists[t];
            bucket.reserve(order.size());
            for (int idx : order) {
                bucket.emplace_back(idx, cost[idx]);
            }
        }

        std::vector<std::vector<std::pair<int,double>>> prefix(d + 1), suffix(d + 1);
        prefix[0] = {{0, 0.0}};
        for (int t = 0; t < d; ++t) {
            prefix[t + 1] = ems_convolve_topk(prefix[t], top_lists[t], K);
        }
        suffix[d] = {{0, 0.0}};
        for (int t = d - 1; t >= 0; --t) {
            suffix[t] = ems_convolve_topk(top_lists[t], suffix[t + 1], K);
        }

        for (int t = 0; t < d; ++t) {
            std::vector<std::pair<int,double>> excl = ems_convolve_topk(prefix[t], suffix[t + 1], K);
            list_to_dense(excl, dense);
            for (int g = 0; g < GF; ++g) {
                cost_out_z[g] = dense[ADDGF[synd][g]];
            }

            double best = std::numeric_limits<double>::infinity();
            for (int g = 0; g < GF; ++g) {
                double c = cost_out_z[g];
                if (std::isfinite(c) && c < best) {
                    best = c;
                }
            }

            std::fill(dest.begin(), dest.end(), 0.0);
            if (std::isfinite(best)) {
                const int a = MatValue[m][t];
                for (int g = 0; g < GF; ++g) {
                    double c = cost_out_z[g];
                    double prob = 0.0;
                    if (std::isfinite(c)) {
                        double delta = c - best;
                        if (delta < 0.0) delta = 0.0;
                        prob = std::exp(-delta);
                    }
                    int gx = DIVGF[g][a];
                    dest[gx] = prob;
                }
            }

            double sum = 0.0;
            for (double &v : dest) {
                if (!std::isfinite(v) || v < 0.0) {
                    v = 0.0;
                }
                sum += v;
            }

            if (sum == 0.0) {
                double uniform = 1.0 / static_cast<double>(GF);
                std::fill(dest.begin(), dest.end(), uniform);
            } else {
                double inv = 1.0 / sum;
                for (double &v : dest) {
                    v *= inv;
                }
            }

            for (int g = 0; g < GF; ++g) {
                CNtoVNxxx[base + t][g] = dest[g];
            }

            normalize_probabilities(CNtoVNxxx[base + t]);
        }
    }
}


#include "check_pass_ems.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

extern std::vector<std::vector<int>> ADDGF;

namespace {

int g_ems_k_override = -1;

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
    int limit,
    int GF) {
    const int aN = static_cast<int>(std::min(A.size(), static_cast<std::size_t>(limit)));
    const int bN = static_cast<int>(std::min(B.size(), static_cast<std::size_t>(limit)));

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
    const int keep = std::min(limit, GF);
    if (static_cast<int>(merged.size()) > keep) {
        merged.resize(keep);
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

void SetCheckPassEMS_K(int override_k) {
    g_ems_k_override = override_k;
}

void ResetCheckPassEMS_K() {
    g_ems_k_override = -1;
}

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
    const int EMS_K = (g_ems_k_override > 0)
                          ? std::min(g_ems_k_override, GF)
                          : std::min(24, GF);
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

        if (d == 1) {
            const int a = MatValue[m][0];
            std::fill(dest.begin(), dest.end(), 0.0);
            if (a == 0) {
                double uniform = 1.0 / static_cast<double>(GF);
                std::fill(dest.begin(), dest.end(), uniform);
            } else {
                int sol = DIVGF[synd][a];
                dest[sol] = 1.0;
            }
            for (int g = 0; g < GF; ++g) {
                CNtoVNxxx[base][g] = dest[g];
            }
            normalize_probabilities(CNtoVNxxx[base]);
            continue;
        }

        std::vector<std::vector<double>> costs(d, std::vector<double>(GF));
        std::vector<std::vector<int>> orders(d, std::vector<int>(GF));
        for (int t = 0; t < d; ++t) {
            const int a = MatValue[m][t];
            for (int g = 0; g < GF; ++g) {
                double p = VNtoCNxxx[base + t][DIVGF[g][a]];
                costs[t][g] = -safe_log(p);
            }
            auto& order = orders[t];
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int lhs, int rhs) { return costs[t][lhs] < costs[t][rhs]; });
        }

        std::vector<std::vector<std::pair<int, double>>> top_lists(d);
        std::vector<std::vector<double>> final_outputs;
        bool success = false;

        const int initial_take = std::min(K, GF);
        // take は EMS で保持する候補数。まず initial_take 個に絞り、
        // その支持集合だけで全てのシンボルに有限コストが割り当てられる
        // （= 制約を満たす組み合わせが一通り揃う）かを確認する。
        // 足りない場合は take を 1 ずつ増やして支持集合を広げ、
        // どのシンボルにも可算なコストが行き渡るまで繰り返す。
        for (int take = initial_take; take <= GF; ++take) {
            const int actual_take = std::min(take, GF);
            for (int t = 0; t < d; ++t) {
                auto& bucket = top_lists[t];
                bucket.clear();
                bucket.reserve(actual_take);
                for (int i = 0; i < actual_take; ++i) {
                    int sym = orders[t][i];
                    bucket.emplace_back(sym, costs[t][sym]);
                }
            }

            std::vector<std::vector<std::pair<int,double>>> prefix(d + 1), suffix(d + 1);
            prefix[0] = {{0, 0.0}};
            for (int t = 0; t < d; ++t) {
                prefix[t + 1] = ems_convolve_topk(prefix[t], top_lists[t], actual_take, GF);
            }
            suffix[d] = {{0, 0.0}};
            for (int t = d - 1; t >= 0; --t) {
                suffix[t] = ems_convolve_topk(top_lists[t], suffix[t + 1], actual_take, GF);
            }

            bool missing_symbol = false;
            std::vector<std::vector<double>> outputs(d, std::vector<double>(GF, 0.0));

            for (int t = 0; t < d; ++t) {
                std::vector<std::pair<int,double>> excl =
                    ems_convolve_topk(prefix[t], suffix[t + 1], actual_take, GF);
                list_to_dense(excl, dense);

                bool edge_missing = false;
                bool any_finite = false;
                bool truncated = false;
                for (int g = 0; g < GF; ++g) {
                    double value = dense[ADDGF[synd][g]];
                    cost_out_z[g] = value;
                    if (std::isfinite(value)) {
                        any_finite = true;
                    } else {
                        truncated = true;
                    }
                }
                if (!any_finite || truncated) {
                    edge_missing = truncated && d > 1;
                    if (!any_finite) {
                        edge_missing = true;
                    }
                }
                if (edge_missing) {
                    missing_symbol = true;
                    break;
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

                outputs[t] = dest;
            }

            if (!missing_symbol) {
                final_outputs = std::move(outputs);
                success = true;
                break;
            }
        }

        if (!success) {
            final_outputs.assign(d, std::vector<double>(GF, 1.0 / static_cast<double>(GF)));
        }

        for (int t = 0; t < d; ++t) {
            for (int g = 0; g < GF; ++g) {
                CNtoVNxxx[base + t][g] = final_outputs[t][g];
            }
            normalize_probabilities(CNtoVNxxx[base + t]);
        }
    }
}


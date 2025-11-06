#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "check_pass_ems.h"

// Global GF tables used by the EMS implementation.
std::vector<std::vector<int>> ADDGF;
std::vector<std::vector<int>> MULGF;
std::vector<std::vector<int>> DIVGF;

namespace {

constexpr int kGF = 8;
constexpr int kNumVars = 16;
constexpr int kRowWeight = 4;
constexpr std::array<std::array<int, kRowWeight>, 8> kCheckVars = {{
    {{0, 1, 2, 3}},     // check 0
    {{4, 5, 6, 7}},     // check 1
    {{8, 9, 10, 11}},   // check 2
    {{12, 13, 14, 15}}, // check 3
    {{0, 4, 8, 12}},    // check 4
    {{1, 5, 9, 13}},    // check 5
    {{2, 6, 10, 14}},   // check 6
    {{3, 7, 11, 15}},   // check 7
}};

void init_gf8_tables() {
    static const int kAdd[kGF][kGF] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 0, 4, 7, 2, 6, 5, 3},
        {2, 4, 0, 5, 1, 3, 7, 6},
        {3, 7, 5, 0, 6, 2, 4, 1},
        {4, 2, 1, 6, 0, 7, 3, 5},
        {5, 6, 3, 2, 7, 0, 1, 4},
        {6, 5, 7, 4, 3, 1, 0, 2},
        {7, 3, 6, 1, 5, 4, 2, 0},
    };

    static const int kMul[kGF][kGF] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 2, 3, 4, 5, 6, 7, 1},
        {0, 3, 4, 5, 6, 7, 1, 2},
        {0, 4, 5, 6, 7, 1, 2, 3},
        {0, 5, 6, 7, 1, 2, 3, 4},
        {0, 6, 7, 1, 2, 3, 4, 5},
        {0, 7, 1, 2, 3, 4, 5, 6},
    };

    static const int kDiv[kGF][kGF] = {
        {-1, 0, 0, 0, 0, 0, 0, 0},
        {-1, 1, 7, 6, 5, 4, 3, 2},
        {-1, 2, 1, 7, 6, 5, 4, 3},
        {-1, 3, 2, 1, 7, 6, 5, 4},
        {-1, 4, 3, 2, 1, 7, 6, 5},
        {-1, 5, 4, 3, 2, 1, 7, 6},
        {-1, 6, 5, 4, 3, 2, 1, 7},
        {-1, 7, 6, 5, 4, 3, 2, 1},
    };

    ADDGF.assign(kGF, std::vector<int>(kGF));
    MULGF.assign(kGF, std::vector<int>(kGF));
    DIVGF.assign(kGF, std::vector<int>(kGF));

    for (int i = 0; i < kGF; ++i) {
        for (int j = 0; j < kGF; ++j) {
            ADDGF[i][j] = kAdd[i][j];
            MULGF[i][j] = kMul[i][j];
            DIVGF[i][j] = kDiv[i][j];
        }
    }
}

void expect_close(const std::vector<double>& actual,
                  const std::vector<double>& expected,
                  double tol = 1e-9) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tol) {
            std::cerr << "Mismatch at index " << i
                      << ": actual=" << actual[i]
                      << ", expected=" << expected[i] << std::endl;
            std::abort();
        }
    }
}

std::vector<int> make_row_bases(const std::vector<int>& row_degree) {
    std::vector<int> bases(row_degree.size() + 1, 0);
    for (std::size_t i = 0; i < row_degree.size(); ++i) {
        bases[i + 1] = bases[i] + row_degree[i];
    }
    return bases;
}

std::vector<int> make_var_symbols() {
    std::vector<int> symbols(kNumVars);
    for (int v = 0; v < kNumVars; ++v) {
        symbols[v] = v % kGF;
    }
    return symbols;
}

void assign_one_hot_messages(const std::vector<int>& row_base,
                             const std::vector<int>& var_symbols,
                             std::vector<std::vector<double>>& VNtoCN) {
    for (std::size_t m = 0; m < kCheckVars.size(); ++m) {
        for (int t = 0; t < kRowWeight; ++t) {
            int edge = row_base[m] + t;
            std::fill(VNtoCN[edge].begin(), VNtoCN[edge].end(), 0.0);
            int var = kCheckVars[m][t];
            VNtoCN[edge][var_symbols[var]] = 1.0;
        }
    }
}

void verify_extrinsic_messages(const std::vector<int>& row_base,
                               const std::vector<int>& var_symbols,
                               const std::vector<int>& syndromes,
                               const std::vector<std::vector<double>>& CNtoVN) {
    for (std::size_t m = 0; m < kCheckVars.size(); ++m) {
        for (int t = 0; t < kRowWeight; ++t) {
            int sum_others = 0;
            for (int k = 0; k < kRowWeight; ++k) {
                if (k == t) continue;
                int var = kCheckVars[m][k];
                sum_others = ADDGF[sum_others][var_symbols[var]];
            }
            int expected_symbol = ADDGF[syndromes[m]][sum_others];
            std::vector<double> expected(kGF, 0.0);
            expected[expected_symbol] = 1.0;
            int edge = row_base[m] + t;
            expect_close(CNtoVN[edge], expected);
        }
    }
}

void run_length16_regular_test(const std::vector<int>& syndromes) {
    init_gf8_tables();

    std::vector<int> RowDegree(kCheckVars.size(), kRowWeight);
    std::vector<std::vector<int>> MatValue(kCheckVars.size(), std::vector<int>(kRowWeight, 1));
    std::vector<int> row_base = make_row_bases(RowDegree);
    int total_edges = row_base.back();

    std::vector<int> var_symbols = make_var_symbols();
    std::vector<std::vector<double>> VNtoCN(total_edges, std::vector<double>(kGF, 0.0));
    std::vector<std::vector<double>> CNtoVN(total_edges, std::vector<double>(kGF, 0.0));

    assign_one_hot_messages(row_base, var_symbols, VNtoCN);

    std::vector<int> TrueNoiseSynd = syndromes;
    CheckPass_EMS(CNtoVN, VNtoCN, MatValue, static_cast<int>(kCheckVars.size()),
                  RowDegree, MULGF, DIVGF, kGF, TrueNoiseSynd);

    verify_extrinsic_messages(row_base, var_symbols, TrueNoiseSynd, CNtoVN);
}

void test_zero_syndrome_length16() {
    run_length16_regular_test(std::vector<int>(kCheckVars.size(), 0));
}

void test_nonzero_syndrome_length16() {
    std::vector<int> syndromes = {1, 2, 3, 4, 5, 6, 7, 1};
    run_length16_regular_test(syndromes);
}

}  // namespace

int main() {
    test_zero_syndrome_length16();
    test_nonzero_syndrome_length16();
    std::cout << "All CheckPass_EMS tests passed." << std::endl;
    return 0;
}

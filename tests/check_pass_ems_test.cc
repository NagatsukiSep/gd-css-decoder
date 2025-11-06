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

void init_gf8_tables() {
    constexpr int GF = 8;
    static const int kAdd[GF][GF] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 0, 4, 7, 2, 6, 5, 3},
        {2, 4, 0, 5, 1, 3, 7, 6},
        {3, 7, 5, 0, 6, 2, 4, 1},
        {4, 2, 1, 6, 0, 7, 3, 5},
        {5, 6, 3, 2, 7, 0, 1, 4},
        {6, 5, 7, 4, 3, 1, 0, 2},
        {7, 3, 6, 1, 5, 4, 2, 0},
    };

    static const int kMul[GF][GF] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 2, 3, 4, 5, 6, 7, 1},
        {0, 3, 4, 5, 6, 7, 1, 2},
        {0, 4, 5, 6, 7, 1, 2, 3},
        {0, 5, 6, 7, 1, 2, 3, 4},
        {0, 6, 7, 1, 2, 3, 4, 5},
        {0, 7, 1, 2, 3, 4, 5, 6},
    };

    static const int kDiv[GF][GF] = {
        {-1, 0, 0, 0, 0, 0, 0, 0},
        {-1, 1, 7, 6, 5, 4, 3, 2},
        {-1, 2, 1, 7, 6, 5, 4, 3},
        {-1, 3, 2, 1, 7, 6, 5, 4},
        {-1, 4, 3, 2, 1, 7, 6, 5},
        {-1, 5, 4, 3, 2, 1, 7, 6},
        {-1, 6, 5, 4, 3, 2, 1, 7},
        {-1, 7, 6, 5, 4, 3, 2, 1},
    };

    ADDGF.assign(GF, std::vector<int>(GF));
    MULGF.assign(GF, std::vector<int>(GF));
    DIVGF.assign(GF, std::vector<int>(GF));

    for (int i = 0; i < GF; ++i) {
        for (int j = 0; j < GF; ++j) {
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

void test_zero_syndrome_weight_four_gf8() {
    init_gf8_tables();

    constexpr int GF = 8;
    std::vector<std::vector<double>> VNtoCN = {
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // x0 = 1
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // x1 = 2
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  // x2 = 3
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  // x3 = 6
    };
    std::vector<std::vector<double>> CNtoVN(4, std::vector<double>(GF, 0.0));
    std::vector<std::vector<int>> MatValue = {{1, 1, 1, 1}};
    std::vector<int> RowDegree = {4};
    std::vector<int> TrueNoiseSynd = {0};

    CheckPass_EMS(CNtoVN, VNtoCN, MatValue, 1, RowDegree, MULGF, DIVGF, GF, TrueNoiseSynd);

    expect_close(CNtoVN[0], std::vector<double>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    expect_close(CNtoVN[1], std::vector<double>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    expect_close(CNtoVN[2], std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    expect_close(CNtoVN[3], std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
}

void test_nonzero_syndrome_weight_four_gf8() {
    init_gf8_tables();

    constexpr int GF = 8;
    std::vector<std::vector<double>> VNtoCN = {
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // x0 = 1
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // x1 = 2
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  // x2 = 3
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  // x3 = 6
    };
    std::vector<std::vector<double>> CNtoVN(4, std::vector<double>(GF, 0.0));
    std::vector<std::vector<int>> MatValue = {{1, 1, 1, 1}};
    std::vector<int> RowDegree = {4};
    std::vector<int> TrueNoiseSynd = {5};

    CheckPass_EMS(CNtoVN, VNtoCN, MatValue, 1, RowDegree, MULGF, DIVGF, GF, TrueNoiseSynd);

    expect_close(CNtoVN[0], std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
    expect_close(CNtoVN[1], std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    expect_close(CNtoVN[2], std::vector<double>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    expect_close(CNtoVN[3], std::vector<double>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}

}  // namespace

int main() {
    test_zero_syndrome_weight_four_gf8();
    test_nonzero_syndrome_weight_four_gf8();
    std::cout << "All CheckPass_EMS tests passed." << std::endl;
    return 0;
}

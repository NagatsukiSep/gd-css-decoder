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

void init_gf2_tables() {
    ADDGF.assign(2, std::vector<int>(2));
    MULGF.assign(2, std::vector<int>(2));
    DIVGF.assign(2, std::vector<int>(2, 0));

    // Addition table (XOR)
    ADDGF[0][0] = 0; ADDGF[0][1] = 1;
    ADDGF[1][0] = 1; ADDGF[1][1] = 0;

    // Multiplication table
    MULGF[0][0] = 0; MULGF[0][1] = 0;
    MULGF[1][0] = 0; MULGF[1][1] = 1;

    // Division by 1 returns the value itself. Division by 0 is unused.
    DIVGF[0][1] = 0;
    DIVGF[1][1] = 1;
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

void test_zero_syndrome_two_variable_check() {
    init_gf2_tables();

    const int GF = 2;
    std::vector<std::vector<double>> VNtoCN = {
        {0.8, 0.2},
        {0.3, 0.7}
    };
    std::vector<std::vector<double>> CNtoVN(2, std::vector<double>(GF, 0.0));
    std::vector<std::vector<int>> MatValue = {{1, 1}};
    std::vector<int> RowDegree = {2};
    std::vector<int> TrueNoiseSynd = {0};

    CheckPass_EMS(CNtoVN, VNtoCN, MatValue, 1, RowDegree, MULGF, DIVGF, GF, TrueNoiseSynd);

    expect_close(CNtoVN[0], std::vector<double>{0.3, 0.7});
    expect_close(CNtoVN[1], std::vector<double>{0.8, 0.2});
}

void test_nonzero_syndrome_two_variable_check() {
    init_gf2_tables();

    const int GF = 2;
    std::vector<std::vector<double>> VNtoCN = {
        {0.8, 0.2},
        {0.3, 0.7}
    };
    std::vector<std::vector<double>> CNtoVN(2, std::vector<double>(GF, 0.0));
    std::vector<std::vector<int>> MatValue = {{1, 1}};
    std::vector<int> RowDegree = {2};
    std::vector<int> TrueNoiseSynd = {1};

    CheckPass_EMS(CNtoVN, VNtoCN, MatValue, 1, RowDegree, MULGF, DIVGF, GF, TrueNoiseSynd);

    expect_close(CNtoVN[0], std::vector<double>{0.7, 0.3});
    expect_close(CNtoVN[1], std::vector<double>{0.2, 0.8});
}

}  // namespace

int main() {
    test_zero_syndrome_two_variable_check();
    test_nonzero_syndrome_two_variable_check();
    std::cout << "All CheckPass_EMS tests passed." << std::endl;
    return 0;
}


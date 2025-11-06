#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "check_pass_ems.h"
#include "check_pass_fft.h"

// Global GF tables used by the EMS implementation.
std::vector<std::vector<int>> ADDGF;
std::vector<std::vector<int>> MULGF;
std::vector<std::vector<int>> DIVGF;

namespace {

constexpr int kGF = 4;
constexpr int kNumVars = 16;
constexpr int kRowWeight = 4;
// 行インデックス m ごとに、接続される変数ノードの列インデックスを列挙した
// 行列。上 4 行は水平方向、下 4 行は垂直方向の制約を表しており、
// 行重み 4・列重み 4 の (16, 8) 形 LDPC 構造になっている。
constexpr std::array<std::array<int, kRowWeight>, 8> kCheckVars = {{
    {{0, 1, 2, 3}},     // チェック 0: 変数 v0〜v3 を束縛
    {{4, 5, 6, 7}},     // チェック 1: 変数 v4〜v7 を束縛
    {{8, 9, 10, 11}},   // チェック 2: 変数 v8〜v11 を束縛
    {{12, 13, 14, 15}}, // チェック 3: 変数 v12〜v15 を束縛
    {{0, 4, 8, 12}},    // チェック 4: 列方向に v0, v4, v8, v12 を束縛
    {{1, 5, 9, 13}},    // チェック 5: 列方向に v1, v5, v9, v13 を束縛
    {{2, 6, 10, 14}},   // チェック 6: 列方向に v2, v6, v10, v14 を束縛
    {{3, 7, 11, 15}},   // チェック 7: 列方向に v3, v7, v11, v15 を束縛
}};

// 各チェック行で用いる係数 a_{m,t} を格納した行列。GF(4) の非零要素を
// バランスよく配置し、係数が 1 のみにならないようにしてある。
constexpr std::array<std::array<int, kRowWeight>, 8> kCheckCoeff = {{
    {{1, 2, 3, 1}},     // 行 0 は 1,2,3,1 を使用
    {{2, 3, 1, 2}},     // 行 1 は 2,3,1,2 を使用
    {{3, 1, 2, 3}},     // 行 2 は 3,1,2,3 を使用
    {{1, 3, 2, 1}},     // 行 3 は 1,3,2,1 を使用
    {{2, 1, 3, 2}},     // 行 4 は 2,1,3,2 を使用
    {{3, 2, 1, 3}},     // 行 5 は 3,2,1,3 を使用
    {{1, 2, 3, 1}},     // 行 6 は 1,2,3,1 を使用
    {{2, 3, 1, 2}},     // 行 7 は 2,3,1,2 を使用
}};

// Walsh–Hadamard 変換の隣接ペアを列挙したテーブル (data/Tables/TENSORFFT4
// の埋め込み版)。CheckPass_FFT の呼び出し時に使用する。
constexpr std::array<std::array<int, 2>, 4> kTensorFFT4 = {{
    {{0, 1}},
    {{2, 3}},
    {{0, 2}},
    {{1, 3}},
}};

void init_gf4_tables() {
    // 既知の GF(4) 加法表を静的配列として保持。ADDGF[a][b] が a+b を返す。
    static const int kAdd[kGF][kGF] = {
        {0, 1, 2, 3},
        {1, 0, 3, 2},
        {2, 3, 0, 1},
        {3, 2, 1, 0},
    };

    // 既知の GF(4) 乗法表。MULGF[a][b] が a×b を返す。
    static const int kMul[kGF][kGF] = {
        {0, 0, 0, 0},
        {0, 1, 2, 3},
        {0, 2, 3, 1},
        {0, 3, 1, 2},
    };

    // 既知の GF(4) 除法表。DIVGF[a][b] が a/b を返す（b=0 は未定義）。
    static const int kDiv[kGF][kGF] = {
        {-1, 0, 0, 0},
        {-1, 1, 3, 2},
        {-1, 2, 1, 3},
        {-1, 3, 2, 1},
    };

    // グローバルテーブルにコピーして、CheckPass_EMS が直接参照できるようにする。
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

// 実際の CN→VN メッセージと、解析的に計算した期待値ベクトルを比較する。
// 1 要素でも許容誤差 tol を超える場合は、ミスマッチ内容を表示して即終了。
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

// 確率ベクトルを「[シンボル:確率, ...]」形式で整形して出力する。
std::string format_distribution(const std::vector<double>& dist) {
    std::ostringstream oss;
    oss << "[";
    oss << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < dist.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << i << ":" << dist[i];
    }
    oss << "]";
    return oss.str();
}

// 各行の開始オフセット（row_base）を構築する。行 m のエッジが
// CN メッセージ配列内のどこから始まるかを表す累積和。
std::vector<int> make_row_bases(const std::vector<int>& row_degree) {
    std::vector<int> bases(row_degree.size() + 1, 0);
    for (std::size_t i = 0; i < row_degree.size(); ++i) {
        bases[i + 1] = bases[i] + row_degree[i];
    }
    return bases;
}

std::vector<std::vector<int>> make_fft_schedule() {
    std::vector<std::vector<int>> schedule(kTensorFFT4.size(),
                                           std::vector<int>(2));
    for (std::size_t i = 0; i < kTensorFFT4.size(); ++i) {
        schedule[i][0] = kTensorFFT4[i][0];
        schedule[i][1] = kTensorFFT4[i][1];
    }
    return schedule;
}

// 各変数ノードが送信したと想定する GF(4) シンボル列。2bit 語 16 個を
// 周期的に散りばめ、行方向・列方向ともに多様な参照値を持つようにする。
constexpr std::array<int, kNumVars> kTransmittedSymbols = {{
    0, 1, 2, 3, 1, 2, 3, 0,  // 行 0〜1 の水平制約で使われる語
    2, 3, 0, 1, 3, 0, 1, 2   // 行 2〜3 および列制約用の語
}};

// 変数ノード 0〜15 それぞれにビット誤り率 0.1 の BSC を仮定した事前確率
// ベクトルを生成する。2bit 語のハミング距離 d に対し、(0.9)^{2-d}(0.1)^d
// を重みとして与え、全体で 1 になるよう正規化する。
std::vector<std::vector<double>> make_soft_messages() {
    std::vector<std::vector<double>> messages(kNumVars,
                                              std::vector<double>(kGF, 0.0));
    constexpr double kBitError = 0.1;
    int bit_width = 0;
    while ((1 << bit_width) < kGF) {
        ++bit_width;
    }

    for (int v = 0; v < kNumVars; ++v) {
        const int transmitted = kTransmittedSymbols[v];
        double total = 0.0;
        for (int symbol = 0; symbol < kGF; ++symbol) {
            int diff = transmitted ^ symbol;
            int distance = 0;
            for (int bit = 0; bit < bit_width; ++bit) {
                if (diff & (1 << bit)) {
                    ++distance;
                }
            }
            double prob = std::pow(1.0 - kBitError, bit_width - distance) *
                          std::pow(kBitError, distance);
            messages[v][symbol] = prob;
            total += prob;
        }
        for (double& value : messages[v]) {
            value /= total;
        }
    }
    return messages;
}

// row_base[m]〜row_base[m+1] のエッジに対し、変数ノードの APP を
// VN→CN メッセージとして複製する（正則行列なので単純コピーでよい）。
void assign_soft_messages(const std::vector<int>& row_base,
                          const std::vector<std::vector<double>>& var_messages,
                          std::vector<std::vector<double>>& VNtoCN) {
    for (std::size_t m = 0; m < kCheckVars.size(); ++m) {
        for (int t = 0; t < kRowWeight; ++t) {
            int edge = row_base[m] + t;
            int var = kCheckVars[m][t];
            VNtoCN[edge] = var_messages[var];
        }
    }
}

// EMS の理論値を得るため、対象エッジ以外の変数を全探索する再帰関数。
//   other_vars : 対象チェックに含まれる「他の」変数のインデックス集合
//   idx        : 現在処理中の other_vars 内位置
//   current_sum: これまでに積み上げた GF 加算結果（z 空間）
//   prob       : その部分割り当てが成立する確率
//   syndrome   : チェック方程式の右辺（制約値）
//   expected   : 解析的に得られる外部情報（CN→VN）を蓄積する配列
void accumulate_expected(const std::vector<std::pair<int, int>>& other_terms,
                         int idx,
                         int current_sum,
                         double prob,
                         const std::vector<std::vector<double>>& var_messages,
                         std::vector<double>& expected_z) {
    if (idx == static_cast<int>(other_terms.size())) {
        expected_z[current_sum] += prob;
        return;
    }

    const auto [var, coeff] = other_terms[idx];
    const auto& message = var_messages[var];
    for (int value = 0; value < kGF; ++value) {
        double next_prob = prob * message[value];
        if (next_prob == 0.0) continue;
        int product = MULGF[coeff][value];
        int next_sum = ADDGF[current_sum][product];
        accumulate_expected(other_terms, idx + 1, next_sum, next_prob,
                            var_messages, expected_z);
    }
}

// 1 行ずつ、EMS 出力 CN→VN が解析解 expected と一致しているかを検証する。
// 各エッジで「自分以外の変数」を列挙して accumulate_expected を呼び出し、
// トータル確率で正規化したのち expect_close で比較する。
void verify_extrinsic_messages(
    const std::vector<int>& row_base,
    const std::vector<std::vector<double>>& var_messages,
    const std::vector<std::vector<double>>& VNtoCN,
    const std::vector<std::vector<int>>& MatValue,
    const std::vector<int>& syndromes,
    const std::vector<std::vector<double>>& CNtoVN,
    const std::vector<std::vector<double>>* CNtoVN_fft) {
    for (std::size_t m = 0; m < kCheckVars.size(); ++m) {
        std::cout << "Check " << m << " update (syndrome " << syndromes[m]
                  << ")" << std::endl;
        for (int t = 0; t < kRowWeight; ++t) {
            std::vector<std::pair<int, int>> other_terms;
            for (int k = 0; k < kRowWeight; ++k) {
                if (k == t) continue;
                other_terms.emplace_back(kCheckVars[m][k], MatValue[m][k]);
            }

            std::vector<double> expected_z(kGF, 0.0);
            accumulate_expected(other_terms, 0, 0, 1.0, var_messages, expected_z);

            std::vector<double> expected_x(kGF, 0.0);
            const int coeff = MatValue[m][t];
            for (int g = 0; g < kGF; ++g) {
                double mass = expected_z[g];
                if (mass == 0.0) continue;
                int z_total = ADDGF[syndromes[m]][g];
                int x_value = DIVGF[z_total][coeff];
                expected_x[x_value] += mass;
            }

            double total = 0.0;
            for (double value : expected_x) {
                total += value;
            }
            if (total > 0.0) {
                for (double& value : expected_x) {
                    value /= total;
                }
            }

            int edge = row_base[m] + t;
            int var = kCheckVars[m][t];
            std::cout << "  Edge to var " << var << " (coeff=" << MatValue[m][t]
                      << ", transmitted=" << kTransmittedSymbols[var] << ")"
                      << std::endl;
            std::cout << "    Input  VN->CN : "
                      << format_distribution(VNtoCN[edge]) << std::endl;
            std::cout << "    Expect CN->VN : "
                      << format_distribution(expected_x) << std::endl;
            std::cout << "    Actual CN->VN : "
                      << format_distribution(CNtoVN[edge]) << std::endl;
            if (CNtoVN_fft) {
                std::cout << "    FFT    CN->VN : "
                          << format_distribution((*CNtoVN_fft)[edge])
                          << std::endl;
                expect_close((*CNtoVN_fft)[edge], expected_x, 1e-6);
            }
            expect_close(CNtoVN[edge], expected_x, 1e-6);
        }
        std::cout << std::endl;
    }
}

// 与えられたシンドローム列に対し、符号長 16・行重み 4・列重み 4 の
// 正則行列で CheckPass_EMS を 1 回実行し、全エッジで期待値照合を行う。
void run_length16_regular_test(const std::vector<int>& syndromes) {
    init_gf4_tables();

    std::vector<int> RowDegree(kCheckVars.size(), kRowWeight);
    std::vector<std::vector<int>> MatValue(kCheckVars.size(), std::vector<int>(kRowWeight, 0));
    for (std::size_t m = 0; m < kCheckVars.size(); ++m) {
        for (int t = 0; t < kRowWeight; ++t) {
            MatValue[m][t] = kCheckCoeff[m][t];
        }
    }
    std::vector<int> row_base = make_row_bases(RowDegree);
    int total_edges = row_base.back();

    std::cout << "Running CheckPass_EMS with syndromes:";
    for (int synd : syndromes) {
        std::cout << ' ' << synd;
    }
    std::cout << std::endl;

    std::vector<std::vector<double>> var_messages = make_soft_messages();
    std::vector<std::vector<double>> VNtoCN(total_edges, std::vector<double>(kGF, 0.0));
    std::vector<std::vector<double>> CNtoVN(total_edges, std::vector<double>(kGF, 0.0));
    std::vector<std::vector<double>> CNtoVN_fft(total_edges, std::vector<double>(kGF, 0.0));

    assign_soft_messages(row_base, var_messages, VNtoCN);

    std::vector<int> TrueNoiseSynd = syndromes;
    CheckPass_EMS(CNtoVN, VNtoCN, MatValue, static_cast<int>(kCheckVars.size()),
                  RowDegree, MULGF, DIVGF, kGF, TrueNoiseSynd);

    auto fft_schedule = make_fft_schedule();
    std::vector<std::vector<double>> VNtoCN_fft = VNtoCN;
    CheckPass_FFT(CNtoVN_fft, VNtoCN_fft, MatValue, static_cast<int>(kCheckVars.size()),
                  RowDegree, MULGF, DIVGF, fft_schedule, kGF, TrueNoiseSynd);

    verify_extrinsic_messages(row_base, var_messages, VNtoCN, MatValue, TrueNoiseSynd,
                              CNtoVN, &CNtoVN_fft);
}

// シンドロームがすべて 0 のケースでは、出力外部情報は純粋に他枝からの
// 合成結果になるはず。全行に対して run_length16_regular_test を適用する。
void test_zero_syndrome_length16() {
    run_length16_regular_test(std::vector<int>(kCheckVars.size(), 0));
}

// 異なる非零シンドロームを各行に割り当て、GF(4) の足し算によるシフトが
// 正しく反映されるかを確認する。
void test_nonzero_syndrome_length16() {
    std::vector<int> syndromes = {1, 2, 3, 1, 2, 3, 1, 2};
    run_length16_regular_test(syndromes);
}

}  // namespace

int main() {
    // ゼロシンドローム・非ゼロシンドロームの 2 ケースを順に実行。
    test_zero_syndrome_length16();
    test_nonzero_syndrome_length16();
    std::cout << "All CheckPass_EMS tests passed." << std::endl;
    return 0;
}


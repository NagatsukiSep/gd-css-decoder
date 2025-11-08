#ifndef GD_CSS_DECODER_CHECK_PASS_FFT_H_
#define GD_CSS_DECODER_CHECK_PASS_FFT_H_

#include <vector>

// FFT (Walsh–Hadamard) ベースの厳密なチェックノード更新を行う参照実装。
// EMS 実装との比較ログを取るため、テストコードから直接呼び出せる
// ように関数シグネチャを公開する。
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
    const std::vector<int>& TrueNoiseSynd);

#endif  // GD_CSS_DECODER_CHECK_PASS_FFT_H_


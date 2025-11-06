#pragma once

#include <vector>

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
);

// EMS で保持する候補数 K を外部から上書きするユーティリティ。
// 負の値を設定するとデフォルトの min(24, GF) が利用される。
void SetCheckPassEMS_K(int override_k);
void ResetCheckPassEMS_K();


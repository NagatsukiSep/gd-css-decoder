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


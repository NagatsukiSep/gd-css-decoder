#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-}

if [[ -z "${BIN}" ]]; then
  if [[ -x "./gd_css_cuda" ]]; then
    BIN=./gd_css_cuda
  else
    BIN=./gd_css
  fi
fi

echo "Running decoder: ${BIN}"

"${BIN}" 500 \
  data/apm_css/DEG_APM_Gamma_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014 \
  data/apm_css/DEG_APM_Delta_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014 \
  DEG_APM_J2_L6_P6500 \
  0.0640 \
  0 \
  101

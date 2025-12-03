#!/usr/bin/env bash
set -euo pipefail


IS_PRINTED_TIME=${1:-0}

DECODE_COUNT=${2:-1}

echo "Running decoder: ./gd_css_cuda"
./gd_css_cuda 500 \
  data/apm_css/DEG_APM_Gamma_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014 \
  data/apm_css/DEG_APM_Delta_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014 \
  DEG_APM_J2_L6_P6500 \
  0.0640 \
  0 \
  101 ${IS_PRINTED_TIME} ${DECODE_COUNT}

echo "✅ Decoding complete"
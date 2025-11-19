#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'USAGE'
Usage: scripts/measure_qbps.sh [options] [decoder-args...]

Options:
  -b, --binary PATH          Decoder binary to execute (defaults to ./gd_css_cuda if
                             present, otherwise ./gd_css)
  -r, --runs N               Number of decoding runs to execute (default: 10)
  -q, --qubits-per-run N     Logical qubits processed per decoder run. When provided,
                             the script prints QBPS (qubits per second).
  -h, --help                 Show this help message and exit.

If no decoder arguments are provided, the script falls back to the same sample
parameters as scripts/run_example.sh.
USAGE
}

BIN=""
RUNS=10
QUBITS_PER_RUN=0
DECODER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--binary)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      BIN="$2"
      shift 2
      ;;
    -r|--runs)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      RUNS="$2"
      shift 2
      ;;
    -q|--qubits-per-run)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      QUBITS_PER_RUN="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    --)
      shift
      DECODER_ARGS+=("$@")
      break
      ;;
    *)
      DECODER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$BIN" ]]; then
  if [[ -x "./gd_css_cuda" ]]; then
    BIN=./gd_css_cuda
  else
    BIN=./gd_css
  fi
fi

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -le 0 ]]; then
  echo "--runs must be a positive integer (got: $RUNS)" >&2
  exit 1
fi

if ! [[ "$QUBITS_PER_RUN" =~ ^[0-9]+$ ]]; then
  echo "--qubits-per-run must be a non-negative integer (got: $QUBITS_PER_RUN)" >&2
  exit 1
fi

if [[ ${#DECODER_ARGS[@]} -eq 0 ]]; then
  DECODER_ARGS=(
    500
    data/apm_css/DEG_APM_Gamma_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014
    data/apm_css/DEG_APM_Delta_J2_L6_P6500_RQ0.333333_alpha2_GF256_GIRTH16_SEED1014
    DEG_APM_J2_L6_P6500
    0.0640
    0
    101
  )
fi

echo "Running decoder: ${BIN}" >&2
echo "Runs: ${RUNS}" >&2
if [[ "$QUBITS_PER_RUN" -gt 0 ]]; then
  echo "Qubits per run: ${QUBITS_PER_RUN}" >&2
fi

TOTAL_NS=0
for ((run=1; run<=RUNS; ++run)); do
  echo "" >&2
  echo "=== Decode run ${run}/${RUNS} ===" >&2
  RUN_START_NS=$(date +%s%N)
  "${BIN}" "${DECODER_ARGS[@]}"
  RUN_END_NS=$(date +%s%N)
  RUN_NS=$((RUN_END_NS - RUN_START_NS))
  TOTAL_NS=$((TOTAL_NS + RUN_NS))
  python3 - "$RUN_NS" <<'PY'
import sys
run_ns = int(sys.argv[1])
run_sec = run_ns / 1e9
print(f"[run] elapsed: {run_sec:.3f} s", file=sys.stderr)
PY
done

python3 - "$TOTAL_NS" "$RUNS" "$QUBITS_PER_RUN" <<'PY'
import sys
import math

total_ns = int(sys.argv[1])
runs = int(sys.argv[2])
qubits_per_run = int(sys.argv[3])

total_sec = total_ns / 1e9
avg_sec = total_sec / runs if runs else float('nan')
print("\n=== QBPS summary ===")
print(f"Total decoding time: {total_sec:.3f} s")
print(f"Average per run:    {avg_sec:.3f} s")
if qubits_per_run > 0 and total_sec > 0:
    total_qubits = qubits_per_run * runs
    qbps = total_qubits / total_sec
    print(f"Processed qubits:   {total_qubits}")
    print(f"QBPS:               {qbps:.3f} qubits/s")
elif qubits_per_run > 0:
    print("Processed qubits:   timing unavailable (total time = 0)")
else:
    print("QBPS:               (provide --qubits-per-run to compute)")
PY

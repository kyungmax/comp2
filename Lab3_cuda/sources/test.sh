#!/usr/bin/env bash
set -uo pipefail

dataset="0"

while [ $# -gt 0 ]; do
  case "$1" in
    -dataset)
      dataset="$2"; shift 2 ;;
    *)
      echo "Usage: $0 [-dataset <number>]"
      exit 1 ;;
  esac
done

echo ">> Testing with dataset ${dataset}"
# If we are not already inside a SLURM allocation, re-invoke this script through
# srun so students do not need to remember the srun wrapper. Compilation and
# execution must run on a compute node (the login node has no GPU).

if [ ! -x ./Convolution_template ]; then
  echo ">> Building template binary"
  make convolution || exit 1
fi

echo ">> Running on Dataset/${dataset}"
log="./test_log_${dataset}.txt"
./Convolution_template \
    -e "./Convolution/Dataset/${dataset}/output.ppm" \
    -i "./Convolution/Dataset/${dataset}/input0.ppm,./Convolution/Dataset/${dataset}/input1.raw" \
    -o "./test_result_${dataset}.ppm" \
    -t image 2>&1 | tee "${log}"

if grep -qiE "solution is correct" "${log}"; then
  echo ">> PASS: dataset ${dataset}"
  exit 0
else
  echo ">> FAIL: dataset ${dataset}"
  exit 1
fi

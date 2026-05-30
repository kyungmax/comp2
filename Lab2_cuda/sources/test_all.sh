#!/usr/bin/env bash
set -euo pipefail

echo ">> Building template binary"
make template

for dataset in {0..14}; do
  echo ">> Running on BatchedMatMul/Dataset/${dataset}"
  ./BatchedMatMul_template \
    -e "./BatchedMatMul/Dataset/${dataset}/output.raw" \
    -i "./BatchedMatMul/Dataset/${dataset}/input0.raw,./BatchedMatMul/Dataset/${dataset}/input1.raw,./BatchedMatMul/Dataset/${dataset}/input2.raw" \
    -o "./test_result_${dataset}.raw" \
    -t matrix
  echo
done

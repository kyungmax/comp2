#!/usr/bin/env bash
set -euo pipefail

dataset="0"

while [ $# -gt 0 ]; do
  case "$1" in
    -dataset)
      dataset="$2"; shift 2;;
    *)
      echo "Usage: $0 -dataset <number>"
      exit 1 ;;
  esac
done

echo ">> Building template binary"
make template
echo ">> Running on BatchedMatMul/Dataset/${dataset}"
./BatchedMatMul_template \
  -e "./BatchedMatMul/Dataset/${dataset}/output.raw" \
  -i "./BatchedMatMul/Dataset/${dataset}/input0.raw,./BatchedMatMul/Dataset/${dataset}/input1.raw,./BatchedMatMul/Dataset/${dataset}/input2.raw" \
  -o ./test_result.raw \
  -t matrix

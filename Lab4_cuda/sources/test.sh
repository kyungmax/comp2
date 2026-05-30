#!/usr/bin/env bash
set -euo pipefail

dataset="0"
build_code_type=""

usage() {
  echo "Usage: $0 -dataset <number> -type <basic|warp|unified>"
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -dataset)
      dataset="$2"; shift 2;;
    -type)
      build_code_type="$2"; shift 2;;
    *)
      usage ;;
  esac
done

# Ensure -type was provided
if [[ -z "$build_code_type" ]]; then
  echo "Error: -type is required."
  usage
fi

echo ">> Building template binary (type=${build_code_type})"
make template_"${build_code_type}"

echo ">> Running on Dataset/${dataset}"
./Reduction_Template \
    -e "./Reduction/Dataset/${dataset}/output.raw" \
    -i "./Reduction/Dataset/${dataset}/input.raw" \
    -o "./test_result_${dataset}.raw" \
    -t vector

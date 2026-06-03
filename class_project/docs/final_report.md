# GPU-Parallel Multi-Entry Upper-Layer Descent for Hard HNSW Queries

## Summary

This project targets hard HNSW queries whose baseline `recall@10 <= 0.5`. The main idea is to expand the upper-layer entry scope, run multiple independent upper-layer descents in parallel on the GPU, select the best resulting base-layer entry point for each query, and then reuse the standard level-0 HNSW search.

On the full GloVe query set, multi-entry upper-layer descent improves hard-query recall substantially. The gain has a clear knee around `entry_count=32` and mostly saturates around `entry_count=64`; `entry_count=260` is the exhaustive top-3-upper-layer pool and adds only small extra recall for much higher cost.

The primary comparison is:

```text
Vanilla Faiss HNSW
vs.
GPU multi-entry upper-layer descent + unchanged CPU level-0 HNSW search
```

This is an optimization of an existing HNSW search pipeline, not a replacement with a full-GPU flat-graph ANN system. The HNSW graph, construction parameters, `efSearch`, and level-0 search are kept fixed; only the upper-layer entry-point selection stage is changed.

## Terminology

- `result_k`: final nearest-neighbor count used by `recall@k`; this report uses `result_k=10`.
- `entry_count`: number of upper-layer entry candidates evaluated before selecting the best base-layer handoff entry.
- `upper_window`: number of highest HNSW upper layers included in the entry candidate pool.
- `hard query`: query whose baseline HNSW `recall@10 <= 0.5` at the same `efSearch`.

## Comparison Scope

The baseline is vanilla Faiss HNSW:

```text
Faiss IndexHNSWFlat
M = 16
efConstruction = 200
efSearch sweep = {16, 32, 64, 128}
standard single-entry hierarchical descent
standard CPU level-0 search
```

The proposed method keeps the same index and the same CPU level-0 search:

```text
same Faiss HNSW graph
same M, efConstruction, and efSearch
GPU-parallel multi-entry upper-layer descent
best base-layer entry handoff
same CPU level-0 search
```

Full-GPU ANN systems such as CAGRA are not used as direct baselines here because they accelerate the entire graph search with a different flat graph structure. That would mostly compare whether the base layer runs on GPU, while this project isolates the effect of optimizing HNSW upper-layer entry selection.

## Experimental Setup

- Dataset: `glove-200-angular.hdf5`
- Index vectors: full train set, `1,183,514 x 200`
- Query vectors: full test set, `10,000 x 200`
- Metric: angular, implemented as normalized inner product in Faiss; exported C++ graph uses equivalent L2 ranking on normalized vectors.
- HNSW parameters:
  - `M=16`
  - `efConstruction=200`
  - `efSearch in {16, 32, 64, 128}`
- Candidate scope:
  - `upper_window=3`
  - top-3 upper-layer candidate pool size: `260`
  - `entry_count in {1, 16, 32, 64, 128, 260}`
- GPU:
  - NVIDIA RTX A6000
  - Driver `565.57.01`
  - CUDA runtime reported by driver: `12.7`
  - `nvcc` version: `12.0`

Generated full-query GPU artifacts are under:

```text
data/faiss_upper_full_m16_efc200_q10000/
data/faiss_upper_full_m16_efc200_q10000/gpu_entries/
```

## Vanilla Faiss HNSW Baseline

The hard query set is recomputed per `efSearch` from the baseline Faiss HNSW result.

| efSearch | recall@10 | hard_n | hard_recall@10 | baseline_search_ms |
|---:|---:|---:|---:|---:|
| 16 | 0.4963 | 5125 | 0.1555 | 145.282 |
| 32 | 0.6066 | 3970 | 0.1873 | 251.934 |
| 64 | 0.7016 | 2994 | 0.2283 | 383.156 |
| 128 | 0.7767 | 2154 | 0.2679 | 694.855 |

## Hard-Query Recall vs. Entry Count

| entry_count | ef=16 | ef=32 | ef=64 | ef=128 |
|---:|---:|---:|---:|---:|
| 1 | 0.1566 | 0.1880 | 0.2282 | 0.2683 |
| 16 | 0.1766 | 0.2030 | 0.2356 | 0.2717 |
| 32 | 0.2566 | 0.2754 | 0.2853 | 0.2936 |
| 64 | 0.2637 | 0.2811 | 0.2880 | 0.2958 |
| 128 | 0.2647 | 0.2835 | 0.2897 | 0.2967 |
| 260 | 0.2657 | 0.2846 | 0.2914 | 0.2990 |

Hard-query recall delta over baseline:

| entry_count | ef=16 | ef=32 | ef=64 | ef=128 |
|---:|---:|---:|---:|---:|
| 16 | +0.0211 | +0.0157 | +0.0073 | +0.0038 |
| 32 | +0.1011 | +0.0881 | +0.0569 | +0.0258 |
| 64 | +0.1082 | +0.0938 | +0.0597 | +0.0279 |
| 128 | +0.1092 | +0.0962 | +0.0614 | +0.0288 |
| 260 | +0.1102 | +0.0973 | +0.0631 | +0.0311 |

The strongest practical jump is from `entry_count=16` to `entry_count=32`. After `entry_count=64`, additional candidates provide only marginal gains. This supports using `entry_count=32` as the latency-conscious setting and `entry_count=64` as the recall-conscious setting.

## Overall Recall

| entry_count | ef=16 | ef=32 | ef=64 | ef=128 |
|---:|---:|---:|---:|---:|
| baseline | 0.4963 | 0.6066 | 0.7016 | 0.7767 |
| 1 | 0.4926 | 0.6020 | 0.6962 | 0.7713 |
| 16 | 0.4983 | 0.6038 | 0.6961 | 0.7706 |
| 32 | 0.5222 | 0.6170 | 0.6986 | 0.7651 |
| 64 | 0.5258 | 0.6182 | 0.6972 | 0.7641 |
| 128 | 0.5269 | 0.6192 | 0.6972 | 0.7638 |
| 260 | 0.5272 | 0.6196 | 0.6977 | 0.7642 |

The method is most beneficial for hard-query recall and low-to-mid `efSearch`. At higher `efSearch`, overall recall can decrease slightly even while hard-query recall improves. This is acceptable for the stated hard-query scope, but it should be reported as a tradeoff rather than hidden.

## GPU Upper-Descent Latency

The graph is uploaded once and kept resident on the GPU. Reported times are for `10,000` queries. The entry files used by the downstream Faiss evaluation were generated from repeat 0.

| entry_count | resident_total_ms | kernel_ms | per_query_us | avg_best_l2 |
|---:|---:|---:|---:|---:|
| 1 | 4.095 | 1.028 | 0.409 | 1.30779 |
| 16 | 31.445 | 5.509 | 3.144 | 1.28789 |
| 32 | 72.127 | 11.049 | 7.213 | 1.17107 |
| 64 | 128.137 | 22.013 | 12.814 | 1.15611 |
| 128 | 250.388 | 46.090 | 25.039 | 1.15085 |
| 260 | 518.192 | 98.550 | 51.819 | 1.14902 |

The CUDA kernel itself scales well with `entry_count`. The larger gap between `resident_total_ms` and `kernel_ms` comes from the current implementation allocating per-call GPU buffers, downloading all `entry_count` descent results, and selecting the best entry on the CPU. A GPU-side best reduction would reduce result download and host-side work.

## End-to-End: Ours vs. Vanilla Faiss

This table combines GPU upper descent with the unchanged CPU Faiss `search_level_0()` from the selected best base-layer entry point. `latency_delta_ms` is measured against vanilla Faiss HNSW at the same `efSearch`.

| entry_count | efSearch | recall@10 | hard_recall@10 | hard_delta | e2e_ms | latency_delta_ms |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 0.5222 | 0.2566 | +0.1011 | 197.884 | +52.601 |
| 32 | 32 | 0.6170 | 0.2754 | +0.0881 | 248.013 | -3.921 |
| 32 | 64 | 0.6986 | 0.2853 | +0.0569 | 356.161 | -26.995 |
| 32 | 128 | 0.7651 | 0.2936 | +0.0258 | 607.710 | -87.145 |
| 64 | 16 | 0.5258 | 0.2637 | +0.1082 | 235.981 | +90.698 |
| 64 | 32 | 0.6182 | 0.2811 | +0.0938 | 296.459 | +44.525 |
| 64 | 64 | 0.6972 | 0.2880 | +0.0597 | 420.137 | +36.981 |
| 64 | 128 | 0.7641 | 0.2958 | +0.0279 | 685.447 | -9.408 |
| 260 | 16 | 0.5272 | 0.2657 | +0.1102 | 611.369 | +466.086 |
| 260 | 32 | 0.6196 | 0.2846 | +0.0973 | 677.880 | +425.946 |
| 260 | 64 | 0.6977 | 0.2914 | +0.0631 | 800.952 | +417.796 |
| 260 | 128 | 0.7642 | 0.2990 | +0.0311 | 1059.586 | +364.731 |

`entry_count=32` has the best recall-latency balance. `entry_count=64` gives slightly higher hard-query recall at a moderate extra cost. `entry_count=260` is useful as an exhaustive upper bound but is not the best operating point.

## Conclusion

The full-query experiment supports the project hypothesis:

1. A meaningful portion of queries are hard under baseline HNSW: `5125 / 10000` queries are hard at `efSearch=16`, and `3970 / 10000` are hard at `efSearch=32`.
2. Expanding the upper-layer entry scope improves hard-query recall. At `entry_count=64`, hard recall improves by `+0.1082` at `efSearch=16` and `+0.0938` at `efSearch=32`.
3. The improvement saturates. `entry_count=32` captures most of the benefit, `entry_count=64` is a robust sweet spot, and exhaustive `entry_count=260` adds only small additional recall.
4. GPU parallelism makes the upper-layer expansion practical. For `10,000` queries, `entry_count=64` takes `128.137 ms` end-to-end in the current GPU upper-descent implementation, with only `22.013 ms` spent in the CUDA kernel.

The recommended final setting is:

```text
M = 16
efConstruction = 200
result_k = 10
upper_window = 3
entry_count = 32 for latency-conscious use
entry_count = 64 for recall-conscious use
handoff = best base-layer entry point
```

## Reproducibility Commands

Build:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CPU/Faiss full-query recall sweep:

```bash
PYTHONPATH=.pydeps python3 scripts/faiss_hnsw_multi_entry.py \
  --dataset data/glove-200-angular.hdf5 \
  --train-size 0 \
  --query-size 10000 \
  --result-k 10 \
  --M 16 \
  --ef-construction 200 \
  --ef-search 16,32,64,128 \
  --faiss-index data/faiss_upper_full_m16_efc200_q200/index.faiss \
  --entry-count 1,16,32,64,128,260 \
  --seed-policy top-layer \
  --upper-window 3 \
  --handoff best \
  --metric angular
```

GPU upper-descent generation:

```bash
./build/bench_gpu_upper_descent \
  data/faiss_upper_full_m16_efc200_q10000 \
  1,16,32,64,128,260 \
  3 \
  3 \
  data/faiss_upper_full_m16_efc200_q10000/gpu_entries
```

GPU-entry downstream evaluation:

```bash
PYTHONPATH=.pydeps python3 scripts/faiss_hnsw_multi_entry.py \
  --dataset data/glove-200-angular.hdf5 \
  --train-size 0 \
  --query-size 10000 \
  --result-k 10 \
  --M 16 \
  --ef-construction 200 \
  --ef-search 16,32,64,128 \
  --faiss-index data/faiss_upper_full_m16_efc200_q200/index.faiss \
  --gpu-entry-dir data/faiss_upper_full_m16_efc200_q10000/gpu_entries \
  --entry-count 1,16,32,64,128,260 \
  --seed-policy none \
  --upper-window 3 \
  --handoff best \
  --metric angular
```

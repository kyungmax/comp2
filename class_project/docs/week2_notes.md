# Week 2 Notes

## Scope Matched to the Proposal

Week 2 in the proposal is:

- implement GPU K-way upper-layer descent;
- map one query to one CUDA block;
- map one independent descent to one warp;
- keep base-layer search unchanged except for accepting K candidate entry points;
- verify CPU/GPU parity for `K in {1, 4, 16}`.

The current code implements that scope as a standalone HNSW-like module:

- CPU greedy upper-layer descent: `greedy_upper_descent`
- CPU K-way descent: `cpu_kway_upper_descent`
- CUDA K-way descent: `gpu_kway_upper_descent`
- deduplicated layer-0 handoff: `entry_points_from_descent` plus `search_layer0`
- parity harness: `week2_parity`

## Mapping From hnswlib

The standalone `UpperLayerGraph` uses CSR arrays so the same graph can be sent to CUDA in one flat representation.

The hnswlib fields map as follows:

- `cur_element_count` -> `UpperLayerGraph::node_count`
- `data_size_ / sizeof(float)` -> `UpperLayerGraph::dim`
- `maxlevel_` -> `UpperLayerGraph::max_level`
- `element_levels_` -> `UpperLayerGraph::node_levels`
- `getDataByInternalId(i)` -> row `i` in `UpperLayerGraph::vectors`
- `get_linklist_at_level(i, level)` -> CSR ranges in `UpperLayerGraph::offsets` and `UpperLayerGraph::neighbors`

For a direct hnswlib integration, add a public or friend exporter that iterates all `level in [0, maxlevel_]` and all internal ids, reads `getListCount(ll)`, and appends each neighbor id to the CSR arrays.

## A6000 Build Target

The RTX A6000 is an Ampere GPU, so the server build should use:

```bash
cmake -S . -B build-cuda -DHNSW_WEEK2_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

If CUDA is enabled, `week2_parity` compares GPU descent output against CPU descent output for every query and every seed at `K = 1, 4, 16`.

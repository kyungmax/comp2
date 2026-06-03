#!/usr/bin/env python3
"""Export a Faiss HNSW graph to the C++ UpperLayerGraph binary format."""

from __future__ import annotations

import argparse
import time
from array import array
from pathlib import Path

import faiss
import h5py
import numpy as np

from faiss_hnsw_multi_entry import build_index, graph_view, normalize_rows


def write_binary(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values)
    with path.open("wb") as f:
        values.tofile(f)


def export_csr(graph, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    n = graph.xb.shape[0]
    max_level = graph.max_level
    cpp_offsets = np.empty(((max_level + 1), n + 1), dtype=np.uint32)
    neighbors_out = array("I")
    cursor = 0

    faiss_offsets = graph.offsets[:n]
    for level in range(max_level + 1):
        cap_begin = int(graph.cum_neighbors[level])
        cap_end = int(graph.cum_neighbors[level + 1])
        cap = cap_end - cap_begin
        cpp_offsets[level, 0] = cursor

        if cap == 0:
            cpp_offsets[level, 1:] = cursor
            continue

        active = graph.levels > level
        active_nodes = np.flatnonzero(active)
        slot_offsets = np.arange(cap_begin, cap_end, dtype=np.int64)
        slots = faiss_offsets[active_nodes, None] + slot_offsets[None, :]
        level_neighbors = graph.neighbors[slots]
        valid = level_neighbors >= 0
        counts = np.zeros(n, dtype=np.uint32)
        counts[active_nodes] = valid.sum(axis=1, dtype=np.uint32)
        cpp_offsets[level, 1:] = cursor + np.cumsum(counts, dtype=np.uint32)

        packed = np.ascontiguousarray(level_neighbors[valid], dtype=np.uint32)
        neighbors_out.frombytes(packed.tobytes())
        cursor += int(packed.size)
        print(
            f"exported level={level} cap={cap} active={int(np.count_nonzero(active))} "
            f"edges={int(packed.size)} total_edges={cursor}",
            flush=True,
        )

    neighbors = np.frombuffer(neighbors_out, dtype=np.uint32)
    return cpp_offsets.reshape(-1), neighbors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/glove-200-angular.hdf5"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/faiss_upper_full_m16_efc200"))
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--query-size", type=int, default=200)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=32)
    parser.add_argument("--metric", choices=["angular", "l2"], default="angular")
    parser.add_argument("--write-index", action="store_true")
    args = parser.parse_args()

    metric = "ip" if args.metric == "angular" else "l2"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with h5py.File(args.dataset, "r") as f:
        train_total = f["train"].shape[0]
        train_size = train_total if args.train_size <= 0 or args.train_size > train_total else args.train_size
        query_size = min(args.query_size, f["test"].shape[0])
        xb = np.asarray(f["train"][:train_size], dtype=np.float32)
        xq = np.asarray(f["test"][:query_size], dtype=np.float32)
    if metric == "ip":
        xb = normalize_rows(xb)
        xq = normalize_rows(xq)
    print(f"loaded train={xb.shape} queries={xq.shape} in {time.perf_counter() - t0:.3f}s", flush=True)

    t0 = time.perf_counter()
    index = build_index(xb, metric, args.M, args.ef_construction, args.ef_search)
    graph = graph_view(index, xb, metric)
    print(
        f"built HNSW in {time.perf_counter() - t0:.3f}s "
        f"max_level={graph.max_level} entry_point={graph.entry_point}",
        flush=True,
    )

    t0 = time.perf_counter()
    offsets, neighbors = export_csr(graph, args.out_dir)
    print(f"converted CSR in {time.perf_counter() - t0:.3f}s neighbors={neighbors.size}", flush=True)

    node_levels = np.ascontiguousarray(graph.levels - 1, dtype=np.int32)
    write_binary(args.out_dir / "vectors.f32", xb.astype(np.float32, copy=False))
    write_binary(args.out_dir / "queries.f32", xq.astype(np.float32, copy=False))
    write_binary(args.out_dir / "node_levels.i32", node_levels)
    write_binary(args.out_dir / "offsets.u32", offsets)
    write_binary(args.out_dir / "neighbors.u32", neighbors)
    if args.write_index:
        t0 = time.perf_counter()
        faiss.write_index(index, str(args.out_dir / "index.faiss"))
        print(f"wrote Faiss index in {time.perf_counter() - t0:.3f}s", flush=True)

    with (args.out_dir / "meta.txt").open("w", encoding="utf-8") as f:
        f.write(f"node_count {xb.shape[0]}\n")
        f.write(f"dim {xb.shape[1]}\n")
        f.write(f"max_level {graph.max_level}\n")
        f.write(f"query_count {xq.shape[0]}\n")
        f.write(f"entry_point {graph.entry_point}\n")
        f.write(f"metric {metric}\n")
    print(f"wrote export to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

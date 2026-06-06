#!/usr/bin/env python3
"""Prototype multi-entry upper-layer descent on top of Faiss HNSW.

This keeps Faiss responsible for graph construction and level-0 HNSW search.
The experiment replaces only the upper-layer entry-point descent:

    Faiss IndexHNSWFlat -> Python upper descent -> Faiss search_level_0()

The same entry-point arrays are the boundary that a CUDA upper-descent kernel
would produce in a C++ integration.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import h5py
import numpy as np


SEED_POLICIES = {
    "none",
    "entry",
    "top",
    "max-level",
    "top-layer",
    "query-top-layer",
    "random-top-layer",
    "random-upper",
    "query-top",
}


@dataclass(frozen=True)
class HnswGraphView:
    xb: np.ndarray
    levels: np.ndarray
    offsets: np.ndarray
    neighbors: np.ndarray
    cum_neighbors: np.ndarray
    entry_point: int
    max_level: int
    metric: str

    def neighbor_ids(self, node: int, level: int) -> np.ndarray:
        if node < 0 or self.levels[node] <= level:
            return np.empty(0, dtype=np.int32)
        begin = self.offsets[node] + self.cum_neighbors[level]
        end = self.offsets[node] + self.cum_neighbors[level + 1]
        ids = self.neighbors[begin:end]
        return ids[ids >= 0]


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def score_batch(queries: np.ndarray, points: np.ndarray, metric: str) -> np.ndarray:
    if metric == "ip":
        return queries @ points.T
    diff = queries[:, None, :] - points[None, :, :]
    return np.sum(diff * diff, axis=2)


def score_one(query: np.ndarray, point: np.ndarray, metric: str) -> float:
    if metric == "ip":
        return float(np.dot(query, point))
    diff = query - point
    return float(np.dot(diff, diff))


def better(candidate: float, current: float, metric: str) -> bool:
    return candidate > current if metric == "ip" else candidate < current


def top_layer_candidates(graph: HnswGraphView, top_layer_window: int) -> np.ndarray:
    if top_layer_window <= 0:
        raise ValueError("top_layer_window must be positive")

    min_faiss_level = max(2, graph.max_level + 2 - top_layer_window)
    nodes = np.flatnonzero(graph.levels >= min_faiss_level).astype(np.int32)
    if nodes.size == 0:
        nodes = np.flatnonzero(graph.levels > 1).astype(np.int32)
    return nodes


def choose_query_top(
    graph: HnswGraphView,
    queries: np.ndarray,
    candidates: np.ndarray,
    entry_k: int,
) -> np.ndarray:
    scores = score_batch(queries, graph.xb[candidates], graph.metric)
    take = min(entry_k, candidates.size)
    if graph.metric == "ip":
        order = np.argpartition(-scores, kth=take - 1, axis=1)
    else:
        order = np.argpartition(scores, kth=take - 1, axis=1)
    chosen = candidates[order[:, :take]]
    if take < entry_k:
        chosen = chosen[:, np.arange(entry_k) % take]
    return np.ascontiguousarray(chosen, dtype=np.int32)


def build_seed_matrix(
    graph: HnswGraphView,
    queries: np.ndarray,
    entry_k: int,
    seed_policy: str,
    rng: np.random.Generator,
    top_layer_window: int,
) -> np.ndarray:
    upper_nodes = np.flatnonzero(graph.levels > 1).astype(np.int32)
    if upper_nodes.size == 0:
        return np.full((queries.shape[0], entry_k), graph.entry_point, dtype=np.int32)

    if seed_policy == "entry":
        return np.full((queries.shape[0], entry_k), graph.entry_point, dtype=np.int32)

    if seed_policy == "top":
        order = np.lexsort((upper_nodes, -graph.levels[upper_nodes]))
        seeds = upper_nodes[order[: min(entry_k, upper_nodes.size)]]
        if seeds.size < entry_k:
            seeds = np.resize(seeds, entry_k)
        return np.tile(seeds[None, :], (queries.shape[0], 1)).astype(np.int32)

    if seed_policy == "max-level":
        max_level_nodes = np.flatnonzero(graph.levels > graph.max_level).astype(np.int32)
        if max_level_nodes.size == 0:
            max_level_nodes = upper_nodes
        seeds = max_level_nodes[: min(entry_k, max_level_nodes.size)]
        if seeds.size < entry_k:
            seeds = np.resize(seeds, entry_k)
        return np.tile(seeds[None, :], (queries.shape[0], 1)).astype(np.int32)

    if seed_policy == "top-layer":
        candidates = top_layer_candidates(graph, top_layer_window)
        order = np.lexsort((candidates, -graph.levels[candidates]))
        seeds = candidates[order[: min(entry_k, candidates.size)]]
        if seeds.size < entry_k:
            seeds = np.resize(seeds, entry_k)
        return np.tile(seeds[None, :], (queries.shape[0], 1)).astype(np.int32)

    if seed_policy == "random-top-layer":
        candidates = top_layer_candidates(graph, top_layer_window)
        seeds = np.empty((queries.shape[0], entry_k), dtype=np.int32)
        replace = candidates.size < entry_k
        for qi in range(queries.shape[0]):
            seeds[qi] = rng.choice(candidates, size=entry_k, replace=replace)
        return seeds

    if seed_policy == "random-upper":
        seeds = np.empty((queries.shape[0], entry_k), dtype=np.int32)
        replace = upper_nodes.size < entry_k
        for qi in range(queries.shape[0]):
            seeds[qi] = rng.choice(upper_nodes, size=entry_k, replace=replace)
        return seeds

    if seed_policy == "query-top-layer":
        candidates = top_layer_candidates(graph, top_layer_window)
        return choose_query_top(graph, queries, candidates, entry_k)

    if seed_policy == "query-top":
        return choose_query_top(graph, queries, upper_nodes, entry_k)

    raise ValueError(f"unknown seed policy: {seed_policy}")


def greedy_upper_descent(
    graph: HnswGraphView,
    query: np.ndarray,
    seed: int,
) -> tuple[int, float, int]:
    current = int(seed)
    current_score = score_one(query, graph.xb[current], graph.metric)
    distance_evals = 1

    start_level = min(graph.max_level, int(graph.levels[current]) - 1)
    for level in range(start_level, 0, -1):
        changed = True
        while changed:
            changed = False
            for nb in graph.neighbor_ids(current, level):
                nb_score = score_one(query, graph.xb[int(nb)], graph.metric)
                distance_evals += 1
                if better(nb_score, current_score, graph.metric):
                    current = int(nb)
                    current_score = nb_score
                    changed = True
    return current, current_score, distance_evals


def multi_entry_upper_descent(
    graph: HnswGraphView,
    queries: np.ndarray,
    seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    nq, entry_k = seeds.shape
    nearest = np.empty((nq, entry_k), dtype=np.int32)
    nearest_d = np.empty((nq, entry_k), dtype=np.float32)
    total_evals = 0
    for qi in range(nq):
        for ki in range(entry_k):
            node, dist, evals = greedy_upper_descent(graph, queries[qi], int(seeds[qi, ki]))
            nearest[qi, ki] = node
            nearest_d[qi, ki] = dist
            total_evals += evals
    return nearest, nearest_d, total_evals


def select_handoff_entries(
    nearest: np.ndarray,
    nearest_d: np.ndarray,
    metric: str,
    handoff: str,
) -> tuple[np.ndarray, np.ndarray]:
    if handoff == "all":
        return nearest, nearest_d

    if handoff != "best":
        raise ValueError(f"unknown handoff policy: {handoff}")

    if metric == "ip":
        best_idx = np.argmax(nearest_d, axis=1)
    else:
        best_idx = np.argmin(nearest_d, axis=1)
    rows = np.arange(nearest.shape[0])
    best_nearest = nearest[rows, best_idx][:, None]
    best_nearest_d = nearest_d[rows, best_idx][:, None]
    return np.ascontiguousarray(best_nearest), np.ascontiguousarray(best_nearest_d)


def call_search_level_0(
    index: faiss.IndexHNSWFlat,
    queries: np.ndarray,
    nearest: np.ndarray,
    nearest_d: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    nearest = np.ascontiguousarray(nearest, dtype=np.int32)
    nearest_d = np.ascontiguousarray(nearest_d, dtype=np.float32)
    distances = np.empty((queries.shape[0], k), dtype=np.float32)
    labels = np.empty((queries.shape[0], k), dtype=np.int64)
    index.search_level_0(
        queries.shape[0],
        faiss.swig_ptr(queries),
        k,
        faiss.swig_ptr(nearest),
        faiss.swig_ptr(nearest_d),
        faiss.swig_ptr(distances),
        faiss.swig_ptr(labels),
        nearest.shape[1],
        2,
    )
    return distances, labels


def load_gpu_entries(
    entry_dir: Path,
    entry_k: int,
    top_layer_window: int,
    query_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    prefix = entry_dir / f"gpu_best_k{entry_k}_w{top_layer_window}"
    nearest = np.fromfile(prefix.with_name(prefix.name + "_nearest.i32"), dtype=np.int32)
    nearest_d = np.fromfile(prefix.with_name(prefix.name + "_nearest_d.f32"), dtype=np.float32)
    if nearest.size != query_count or nearest_d.size != query_count:
        raise ValueError(
            f"GPU entry file size mismatch for entry_count={entry_k}, window={top_layer_window}: "
            f"nearest={nearest.size}, nearest_d={nearest_d.size}, query_count={query_count}"
        )
    meta = read_meta(prefix.with_name(prefix.name + "_meta.txt"))
    return nearest[:, None], nearest_d[:, None], meta


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    per_query = recall_per_query(found, truth, k)
    return float(np.mean(per_query)) if per_query.size > 0 else 0.0


def recall_per_query(found: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    recalls = np.empty(found.shape[0], dtype=np.float32)
    for i, (got, expected) in enumerate(zip(found[:, :k], truth[:, :k])):
        hits = len(set(map(int, got)) & set(map(int, expected)))
        recalls[i] = hits / float(k)
    return recalls


def recall_at_k_masked(found: np.ndarray, truth: np.ndarray, k: int, mask: np.ndarray) -> float:
    if int(np.count_nonzero(mask)) == 0:
        return float("nan")
    return float(np.mean(recall_per_query(found[mask], truth[mask], k)))


def exact_ground_truth(xb: np.ndarray, xq: np.ndarray, k: int, metric: str) -> np.ndarray:
    exact = faiss.IndexFlatIP(xb.shape[1]) if metric == "ip" else faiss.IndexFlatL2(xb.shape[1])
    exact.add(xb)
    _, labels = exact.search(xq, k)
    return labels


def load_hdf5(path: Path, train_size: int, query_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        train_total = f["train"].shape[0]
        if train_size <= 0 or train_size > train_total:
            train_size = train_total
        query_total = f["test"].shape[0]
        query_size = min(query_size, query_total)
        xb = np.asarray(f["train"][:train_size], dtype=np.float32)
        xq = np.asarray(f["test"][:query_size], dtype=np.float32)
        truth = None
        if train_size == train_total and "neighbors" in f:
            truth = np.asarray(f["neighbors"][:query_size], dtype=np.int64)
    return xb, xq, truth


def build_index(xb: np.ndarray, metric: str, m: int, ef_construction: int, ef_search: int) -> faiss.IndexHNSWFlat:
    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(xb.shape[1], m, faiss_metric)
    index.hnsw.efConstruction = ef_construction
    index.add(xb)
    index.hnsw.efSearch = ef_search
    return index


def graph_view(index: faiss.IndexHNSWFlat, xb: np.ndarray, metric: str) -> HnswGraphView:
    hnsw = index.hnsw
    return HnswGraphView(
        xb=xb,
        levels=faiss.vector_to_array(hnsw.levels).astype(np.int32, copy=False),
        offsets=faiss.vector_to_array(hnsw.offsets).astype(np.int64, copy=False),
        neighbors=faiss.vector_to_array(hnsw.neighbors).astype(np.int32, copy=False),
        cum_neighbors=faiss.vector_to_array(hnsw.cum_nneighbor_per_level).astype(np.int64, copy=False),
        entry_point=int(hnsw.entry_point),
        max_level=int(hnsw.max_level),
        metric=metric,
    )


def level_summary(graph: HnswGraphView) -> str:
    unique_levels = np.unique(graph.levels)
    hist = ",".join(
        f"{int(level)}:{int(np.count_nonzero(graph.levels == level))}"
        for level in unique_levels
    )
    upper_count = int(np.count_nonzero(graph.levels > 1))
    max_level_count = int(np.count_nonzero(graph.levels > graph.max_level))
    return (
        f"upper_nodes={upper_count} max_level_nodes={max_level_count} "
        f"level_hist={hist}"
    )


def top_layer_summary(graph: HnswGraphView, top_layer_window: int) -> str:
    candidates = top_layer_candidates(graph, top_layer_window)
    min_faiss_level = max(2, graph.max_level + 2 - top_layer_window)
    return (
        f"top_layer_window={top_layer_window} "
        f"min_faiss_level={min_faiss_level} "
        f"top_layer_candidates={candidates.size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/glove-200-angular.hdf5"))
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--query-size", type=int, default=200)
    parser.add_argument("--result-k", "--k", dest="k", metavar="RESULT_K", type=int, default=10)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=parse_csv_ints, default=[32])
    parser.add_argument("--faiss-index", type=Path)
    parser.add_argument("--gpu-entry-dir", type=Path)
    parser.add_argument(
        "--entry-count",
        "--entry-k",
        dest="entry_k",
        metavar="ENTRY_COUNT",
        type=parse_csv_ints,
        default=[1, 4, 16],
        help="Comma-separated upper-layer entry candidate counts; this is not result-k",
    )
    parser.add_argument(
        "--seed-policy",
        default="top",
        help=(
            "Comma-separated policies from: "
            "none,entry,top,max-level,top-layer,query-top-layer,random-top-layer,random-upper,query-top"
        ),
    )
    parser.add_argument(
        "--upper-window",
        "--top-layer-window",
        dest="top_layer_window",
        metavar="UPPER_WINDOW",
        type=parse_csv_ints,
        default=[2],
        help="Comma-separated counts of highest HNSW layers to include for *-top-layer seed policies",
    )
    parser.add_argument("--handoff", choices=["best", "all"], default="best")
    parser.add_argument(
        "--hard-recall-max",
        type=float,
        default=0.5,
        help="Queries with baseline per-query recall@result_k <= this value are reported as hard",
    )
    parser.add_argument("--metric", choices=["angular", "l2"], default="angular")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metric = "ip" if args.metric == "angular" else "l2"
    rng = np.random.default_rng(args.seed)
    seed_policies = [part.strip() for part in args.seed_policy.split(",") if part.strip()]
    invalid_policies = sorted(set(seed_policies) - SEED_POLICIES)
    if invalid_policies:
        raise ValueError(f"unknown seed policies: {','.join(invalid_policies)}")
    if not args.top_layer_window or any(window <= 0 for window in args.top_layer_window):
        raise ValueError("top-layer-window values must be positive")
    if not args.ef_search or any(ef <= 0 for ef in args.ef_search):
        raise ValueError("ef-search values must be positive")

    t0 = time.perf_counter()
    xb, xq, truth = load_hdf5(args.dataset, args.train_size, args.query_size)
    if metric == "ip":
        xb = normalize_rows(xb)
        xq = normalize_rows(xq)
    load_s = time.perf_counter() - t0

    if truth is None:
        t0 = time.perf_counter()
        truth = exact_ground_truth(xb, xq, args.k, metric)
        gt_s = time.perf_counter() - t0
    else:
        gt_s = 0.0

    t0 = time.perf_counter()
    if args.faiss_index is None:
        index = build_index(xb, metric, args.M, args.ef_construction, args.ef_search[0])
    else:
        index = faiss.read_index(str(args.faiss_index))
        index.hnsw.efSearch = args.ef_search[0]
    build_s = time.perf_counter() - t0
    graph = graph_view(index, xb, metric)

    print(f"dataset={args.dataset}")
    print(f"train={xb.shape} queries={xq.shape} metric={metric} result_k={args.k}")
    print(
        f"M={args.M} efConstruction={args.ef_construction} efSearch={','.join(map(str, args.ef_search))} "
        f"max_level={graph.max_level} entry_point={graph.entry_point}"
    )
    print(level_summary(graph))
    for top_layer_window in args.top_layer_window:
        print(top_layer_summary(graph, top_layer_window))
    print(f"load_s={load_s:.3f} gt_s={gt_s:.3f} build_s={build_s:.3f}")
    if args.faiss_index is not None:
        print(f"loaded_faiss_index={args.faiss_index}")

    hard_masks: dict[int, np.ndarray] = {}
    baseline_hard_recalls: dict[int, float] = {}
    baseline_search_ms: dict[int, float] = {}
    for ef_search in args.ef_search:
        index.hnsw.efSearch = ef_search
        t0 = time.perf_counter()
        _, baseline_labels = index.search(xq, args.k)
        baseline_s = time.perf_counter() - t0
        baseline_search_ms[ef_search] = baseline_s * 1000.0
        baseline_per_query = recall_per_query(baseline_labels, truth, args.k)
        baseline_recall = float(np.mean(baseline_per_query))
        hard_mask = baseline_per_query <= args.hard_recall_max
        hard_masks[ef_search] = hard_mask
        hard_count = int(np.count_nonzero(hard_mask))
        hard_recall = float(np.mean(baseline_per_query[hard_mask])) if hard_count > 0 else float("nan")
        baseline_hard_recalls[ef_search] = hard_recall
        print(
            f"faiss_baseline efSearch={ef_search} "
            f"recall@{args.k}={baseline_recall:.4f} "
            f"hard_n={hard_count} hard_recall@{args.k}={hard_recall:.4f} "
            f"search_ms={baseline_s * 1000:.3f}"
        )
    print(
        "efSearch upper_window entry_count seed_policy handoff upper_ms level0_ms e2e_ms latency_delta_ms "
        "recall hard_n hard_recall hard_delta evals_per_query entries_per_query"
    )

    for top_layer_window in args.top_layer_window:
        for seed_policy in seed_policies:
            if seed_policy == "none":
                continue
            for entry_k in args.entry_k:
                seeds = build_seed_matrix(
                    graph,
                    xq,
                    entry_k,
                    seed_policy,
                    rng,
                    top_layer_window,
                )
                t0 = time.perf_counter()
                nearest, nearest_d, evals = multi_entry_upper_descent(graph, xq, seeds)
                upper_s = time.perf_counter() - t0

                handoff_nearest, handoff_nearest_d = select_handoff_entries(
                    nearest,
                    nearest_d,
                    metric,
                    args.handoff,
                )
                for ef_search in args.ef_search:
                    index.hnsw.efSearch = ef_search
                    t0 = time.perf_counter()
                    _, labels = call_search_level_0(
                        index,
                        xq,
                        handoff_nearest,
                        handoff_nearest_d,
                        args.k,
                    )
                    level0_s = time.perf_counter() - t0

                    rec = recall_at_k(labels, truth, args.k)
                    hard_mask = hard_masks[ef_search]
                    hard_count = int(np.count_nonzero(hard_mask))
                    hard_recall = recall_at_k_masked(labels, truth, args.k, hard_mask)
                    hard_delta = hard_recall - baseline_hard_recalls[ef_search]
                    e2e_ms = upper_s * 1000.0 + level0_s * 1000.0
                    print(
                        f"{ef_search:8d} {top_layer_window:6d} {entry_k:7d} "
                        f"{seed_policy:15s} {args.handoff:7s} "
                        f"{upper_s * 1000:8.3f} {level0_s * 1000:9.3f} "
                        f"{e2e_ms:8.3f} {e2e_ms - baseline_search_ms[ef_search]:16.3f} "
                        f"{rec:.4f} {hard_count:d} {hard_recall:.4f} "
                        f"{hard_delta:.4f} "
                        f"{evals / xq.shape[0]:.1f} {handoff_nearest.shape[1]:.1f}"
                    )

    if args.gpu_entry_dir is not None:
        for top_layer_window in args.top_layer_window:
            for entry_k in args.entry_k:
                handoff_nearest, handoff_nearest_d, gpu_meta = load_gpu_entries(
                    args.gpu_entry_dir,
                    entry_k,
                    top_layer_window,
                    xq.shape[0],
                )
                upper_ms = float(gpu_meta.get("resident_total_ms", "nan"))
                for ef_search in args.ef_search:
                    index.hnsw.efSearch = ef_search
                    t0 = time.perf_counter()
                    _, labels = call_search_level_0(
                        index,
                        xq,
                        handoff_nearest,
                        handoff_nearest_d,
                        args.k,
                    )
                    level0_s = time.perf_counter() - t0

                    rec = recall_at_k(labels, truth, args.k)
                    hard_mask = hard_masks[ef_search]
                    hard_count = int(np.count_nonzero(hard_mask))
                    hard_recall = recall_at_k_masked(labels, truth, args.k, hard_mask)
                    hard_delta = hard_recall - baseline_hard_recalls[ef_search]
                    e2e_ms = upper_ms + level0_s * 1000.0
                    print(
                        f"{ef_search:8d} {top_layer_window:6d} {entry_k:7d} "
                        f"{'gpu-top-layer':15s} {args.handoff:7s} "
                        f"{upper_ms:8.3f} {level0_s * 1000:9.3f} "
                        f"{e2e_ms:8.3f} {e2e_ms - baseline_search_ms[ef_search]:16.3f} "
                        f"{rec:.4f} {hard_count:d} {hard_recall:.4f} "
                        f"{hard_delta:.4f} "
                        f"nan {handoff_nearest.shape[1]:.1f}"
                    )


if __name__ == "__main__":
    main()

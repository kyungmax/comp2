#include "hnsw_week2/gpu_upper_descent.hpp"
#include "hnsw_week2/upper_descent.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using hnsw_week2::DescentResult;
using hnsw_week2::NodeId;
using hnsw_week2::SearchResult;
using hnsw_week2::UpperLayerGraph;

namespace {

std::vector<float> make_centers(std::size_t clusters, std::size_t dim) {
    std::vector<float> centers(clusters * dim, 0.0f);
    for (std::size_t c = 0; c < clusters; ++c) {
        centers[c * dim + (c % dim)] = static_cast<float>(8 + c * 3);
        centers[c * dim + ((c + 3) % dim)] = -static_cast<float>(4 + c);
    }
    return centers;
}

void append_knn_edges(
    const std::vector<float>& vectors,
    std::size_t dim,
    const std::vector<int>& node_levels,
    int level,
    std::size_t degree,
    std::vector<std::vector<NodeId>>& adjacency) {
    const std::size_t n = node_levels.size();
    for (NodeId u = 0; u < n; ++u) {
        if (node_levels[u] < level) continue;

        std::vector<std::pair<float, NodeId>> distances;
        for (NodeId v = 0; v < n; ++v) {
            if (u == v || node_levels[v] < level) continue;
            distances.push_back({
                hnsw_week2::l2_squared(
                    vectors.data() + static_cast<std::size_t>(u) * dim,
                    vectors.data() + static_cast<std::size_t>(v) * dim,
                    dim),
                v});
        }
        std::sort(distances.begin(), distances.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) return lhs.second < rhs.second;
            return lhs.first < rhs.first;
        });

        const std::size_t take = std::min(degree, distances.size());
        adjacency[u].reserve(take);
        for (std::size_t i = 0; i < take; ++i) {
            adjacency[u].push_back(distances[i].second);
        }
    }
}

UpperLayerGraph make_test_graph() {
    constexpr std::size_t clusters = 4;
    constexpr std::size_t points_per_cluster = 16;
    constexpr std::size_t dim = 8;
    constexpr std::size_t n = clusters * points_per_cluster;
    constexpr int max_level = 2;

    UpperLayerGraph graph;
    graph.node_count = n;
    graph.dim = dim;
    graph.max_level = max_level;
    graph.vectors.resize(n * dim);
    graph.node_levels.resize(n);

    const std::vector<float> centers = make_centers(clusters, dim);
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.18f);

    for (NodeId node = 0; node < n; ++node) {
        const std::size_t cluster = node / points_per_cluster;
        for (std::size_t d = 0; d < dim; ++d) {
            graph.vectors[static_cast<std::size_t>(node) * dim + d] =
                centers[cluster * dim + d] + noise(rng);
        }

        if (node % 4 == 0) {
            graph.node_levels[node] = 2;
        } else if (node % 2 == 0) {
            graph.node_levels[node] = 1;
        } else {
            graph.node_levels[node] = 0;
        }
    }

    std::vector<std::vector<std::vector<NodeId>>> adjacency_by_level(
        max_level + 1,
        std::vector<std::vector<NodeId>>(n));
    append_knn_edges(graph.vectors, graph.dim, graph.node_levels, 0, 10, adjacency_by_level[0]);
    append_knn_edges(graph.vectors, graph.dim, graph.node_levels, 1, 6, adjacency_by_level[1]);
    append_knn_edges(graph.vectors, graph.dim, graph.node_levels, 2, 4, adjacency_by_level[2]);

    graph.offsets.resize(static_cast<std::size_t>(max_level + 1) * (n + 1), 0);
    for (int level = 0; level <= max_level; ++level) {
        const std::size_t base = static_cast<std::size_t>(level) * (n + 1);
        graph.offsets[base] = static_cast<std::uint32_t>(graph.neighbors.size());
        for (NodeId node = 0; node < n; ++node) {
            for (NodeId neighbor : adjacency_by_level[level][node]) {
                graph.neighbors.push_back(neighbor);
            }
            graph.offsets[base + node + 1] = static_cast<std::uint32_t>(graph.neighbors.size());
        }
    }

    graph.validate();
    return graph;
}

std::vector<float> make_queries(std::size_t dim) {
    constexpr std::size_t clusters = 4;
    std::vector<float> centers = make_centers(clusters, dim);
    for (std::size_t c = 0; c < clusters; ++c) {
        for (std::size_t d = 0; d < dim; ++d) {
            centers[c * dim + d] += 0.05f * static_cast<float>(static_cast<int>(d % 3) - 1);
        }
    }
    return centers;
}

std::vector<NodeId> exact_knn(const UpperLayerGraph& graph, const float* query, std::size_t top_k) {
    std::vector<std::pair<float, NodeId>> distances;
    distances.reserve(graph.node_count);
    for (NodeId node = 0; node < graph.node_count; ++node) {
        distances.push_back({
            hnsw_week2::l2_squared(
                query,
                graph.vectors.data() + static_cast<std::size_t>(node) * graph.dim,
                graph.dim),
            node});
    }
    std::sort(distances.begin(), distances.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first == rhs.first) return lhs.second < rhs.second;
        return lhs.first < rhs.first;
    });

    std::vector<NodeId> result;
    for (std::size_t i = 0; i < std::min(top_k, distances.size()); ++i) {
        result.push_back(distances[i].second);
    }
    return result;
}

double recall_at(const std::vector<SearchResult>& approximate, const std::vector<NodeId>& exact) {
    std::set<NodeId> exact_set(exact.begin(), exact.end());
    std::size_t hits = 0;
    for (const SearchResult& result : approximate) {
        if (exact_set.count(result.node) != 0) ++hits;
    }
    return exact.empty() ? 1.0 : static_cast<double>(hits) / static_cast<double>(exact.size());
}

float best_entry_distance(const std::vector<DescentResult>& descents) {
    float best = std::numeric_limits<float>::infinity();
    for (const DescentResult& descent : descents) {
        best = std::min(best, descent.entry_distance);
    }
    return best;
}

std::vector<NodeId> interleaved_top_seeds(const UpperLayerGraph& graph) {
    std::vector<NodeId> seeds = hnsw_week2::top_level_seeds(graph, 16);
    std::sort(seeds.begin(), seeds.end(), [](NodeId lhs, NodeId rhs) {
        const NodeId lhs_position_in_cluster = lhs % 16;
        const NodeId rhs_position_in_cluster = rhs % 16;
        if (lhs_position_in_cluster == rhs_position_in_cluster) return lhs < rhs;
        return lhs_position_in_cluster < rhs_position_in_cluster;
    });
    return seeds;
}

void check_cpu_path(const UpperLayerGraph& graph, const std::vector<float>& queries) {
    const std::vector<NodeId> seeds = interleaved_top_seeds(graph);
    assert(seeds.size() >= 16);

    const std::vector<std::size_t> ks = {1, 4, 16};
    for (std::size_t q = 0; q < queries.size() / graph.dim; ++q) {
        const float* query = queries.data() + q * graph.dim;
        float previous_best = std::numeric_limits<float>::infinity();

        for (std::size_t k : ks) {
            const std::vector<DescentResult> descents =
                hnsw_week2::cpu_kway_upper_descent(graph, query, seeds, k);
            assert(descents.size() == k);
            const float best_distance = best_entry_distance(descents);
            assert(best_distance <= previous_best + 1e-5f);
            previous_best = best_distance;

            const std::vector<NodeId> entry_points =
                hnsw_week2::entry_points_from_descent(descents);
            const std::vector<SearchResult> approximate =
                hnsw_week2::search_layer0(graph, query, entry_points, 24, 5);
            const std::vector<NodeId> best_entry = {
                hnsw_week2::best_entry_point_from_descent(descents)};
            const std::vector<SearchResult> best_entry_approximate =
                hnsw_week2::search_layer0(graph, query, best_entry, 24, 5);
            const std::vector<NodeId> exact = exact_knn(graph, query, 5);
            const double recall = recall_at(approximate, exact);
            const double best_entry_recall = recall_at(best_entry_approximate, exact);
            assert(!approximate.empty());
            assert(!best_entry_approximate.empty());
            std::cout << "cpu q=" << q << " entry_count=" << k
                      << " best_ep_dist=" << best_distance
                      << " all_entry_recall@5=" << recall
                      << " best_entry_recall@5=" << best_entry_recall << '\n';
        }
    }
}

void check_gpu_path(const UpperLayerGraph& graph, const std::vector<float>& queries) {
    if (!hnsw_week2::cuda_available()) {
        std::cout << "cuda unavailable: skipped GPU parity checks\n";
        return;
    }

    const std::vector<NodeId> seeds = interleaved_top_seeds(graph);
    const std::size_t query_count = queries.size() / graph.dim;

    for (std::size_t k : {std::size_t{1}, std::size_t{4}, std::size_t{16}, std::size_t{40}}) {
        std::vector<NodeId> seeds_for_k(k);
        for (std::size_t i = 0; i < k; ++i) {
            seeds_for_k[i] = seeds[i % seeds.size()];
        }
        std::vector<NodeId> batched_seeds(query_count * k);
        for (std::size_t q = 0; q < query_count; ++q) {
            std::copy_n(seeds_for_k.begin(), k, batched_seeds.begin() + q * k);
        }

        const std::vector<DescentResult> gpu_descents =
            hnsw_week2::gpu_kway_upper_descent(graph, queries, query_count, batched_seeds, k);
        assert(gpu_descents.size() == query_count * k);

        for (std::size_t q = 0; q < query_count; ++q) {
            const float* query = queries.data() + q * graph.dim;
            const std::vector<DescentResult> cpu_descents =
                hnsw_week2::cpu_kway_upper_descent(graph, query, seeds_for_k, k);

            for (std::size_t i = 0; i < k; ++i) {
                const DescentResult& cpu = cpu_descents[i];
                const DescentResult& gpu = gpu_descents[q * k + i];
                assert(cpu.seed == gpu.seed);
                assert(cpu.entry_point == gpu.entry_point);
                assert(std::fabs(cpu.entry_distance - gpu.entry_distance) < 1e-4f);
            }

            const std::vector<NodeId> cpu_entries =
                hnsw_week2::entry_points_from_descent(cpu_descents);
            const std::vector<NodeId> cpu_best_entry = {
                hnsw_week2::best_entry_point_from_descent(cpu_descents)};
            const std::vector<DescentResult> gpu_slice(
                gpu_descents.begin() + q * k,
                gpu_descents.begin() + (q + 1) * k);
            const std::vector<NodeId> gpu_entries =
                hnsw_week2::entry_points_from_descent(gpu_slice);
            const std::vector<NodeId> gpu_best_entry = {
                hnsw_week2::best_entry_point_from_descent(gpu_slice)};
            const std::vector<SearchResult> cpu_results =
                hnsw_week2::search_layer0(graph, query, cpu_entries, 24, 5);
            const std::vector<SearchResult> gpu_results =
                hnsw_week2::search_layer0(graph, query, gpu_entries, 24, 5);
            const std::vector<SearchResult> cpu_best_entry_results =
                hnsw_week2::search_layer0(graph, query, cpu_best_entry, 24, 5);
            const std::vector<SearchResult> gpu_best_entry_results =
                hnsw_week2::search_layer0(graph, query, gpu_best_entry, 24, 5);
            assert(cpu_results.size() == gpu_results.size());
            for (std::size_t i = 0; i < cpu_results.size(); ++i) {
                assert(cpu_results[i].node == gpu_results[i].node);
            }
            assert(cpu_best_entry_results.size() == gpu_best_entry_results.size());
            for (std::size_t i = 0; i < cpu_best_entry_results.size(); ++i) {
                assert(cpu_best_entry_results[i].node == gpu_best_entry_results[i].node);
            }
        }
    }
}

}  // namespace

int main() {
    const UpperLayerGraph graph = make_test_graph();
    const std::vector<float> queries = make_queries(graph.dim);

    check_cpu_path(graph, queries);
    check_gpu_path(graph, queries);

    std::cout << "week2 parity checks passed\n";
    return 0;
}

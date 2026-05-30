#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace hnsw_week2 {

using NodeId = std::uint32_t;
constexpr NodeId kInvalidNode = std::numeric_limits<NodeId>::max();

struct UpperLayerGraph {
    std::size_t node_count = 0;
    std::size_t dim = 0;
    int max_level = 0;

    std::vector<float> vectors;
    std::vector<int> node_levels;

    // CSR adjacency for levels [0, max_level].
    // offsets[level * (node_count + 1) + node] gives the start in neighbors.
    std::vector<std::uint32_t> offsets;
    std::vector<NodeId> neighbors;

    std::size_t level_stride() const;
    std::pair<const NodeId*, const NodeId*> neighbor_range(int level, NodeId node) const;
    bool has_node_at_level(NodeId node, int level) const;
    void validate() const;
};

struct DescentResult {
    NodeId seed = kInvalidNode;
    NodeId entry_point = kInvalidNode;
    float entry_distance = std::numeric_limits<float>::infinity();
    int start_level = 0;
    std::size_t distance_evaluations = 0;
    std::size_t hops = 0;
};

struct SearchResult {
    NodeId node = kInvalidNode;
    float distance = std::numeric_limits<float>::infinity();
};

float l2_squared(const float* a, const float* b, std::size_t dim);

DescentResult greedy_upper_descent(
    const UpperLayerGraph& graph,
    const float* query,
    NodeId seed,
    int start_level);

std::vector<DescentResult> cpu_kway_upper_descent(
    const UpperLayerGraph& graph,
    const float* query,
    const std::vector<NodeId>& seeds,
    std::size_t k);

std::vector<SearchResult> search_layer0(
    const UpperLayerGraph& graph,
    const float* query,
    const std::vector<NodeId>& entry_points,
    std::size_t ef,
    std::size_t top_k);

std::vector<NodeId> top_level_seeds(const UpperLayerGraph& graph, std::size_t max_count);

std::vector<NodeId> entry_points_from_descent(const std::vector<DescentResult>& descents);

}  // namespace hnsw_week2

#include "hnsw_week2/upper_descent.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace hnsw_week2 {

namespace {

struct Candidate {
    float distance;
    NodeId node;
};

struct MinCandidate {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const {
        if (lhs.distance == rhs.distance) return lhs.node > rhs.node;
        return lhs.distance > rhs.distance;
    }
};

struct MaxCandidate {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const {
        if (lhs.distance == rhs.distance) return lhs.node < rhs.node;
        return lhs.distance < rhs.distance;
    }
};

void ensure_query(const float* query) {
    if (query == nullptr) {
        throw std::invalid_argument("query pointer must not be null");
    }
}

}  // namespace

std::size_t UpperLayerGraph::level_stride() const {
    return node_count + 1;
}

std::pair<const NodeId*, const NodeId*> UpperLayerGraph::neighbor_range(int level, NodeId node) const {
    if (level < 0 || level > max_level || node >= node_count) {
        throw std::out_of_range("invalid level or node");
    }
    const std::size_t base = static_cast<std::size_t>(level) * level_stride();
    const std::uint32_t begin = offsets[base + node];
    const std::uint32_t end = offsets[base + node + 1];
    if (begin == end) {
        return {nullptr, nullptr};
    }
    return {neighbors.data() + begin, neighbors.data() + end};
}

bool UpperLayerGraph::has_node_at_level(NodeId node, int level) const {
    return node < node_count && level >= 0 && level <= max_level && node_levels[node] >= level;
}

void UpperLayerGraph::validate() const {
    if (node_count == 0) {
        throw std::invalid_argument("graph must contain at least one node");
    }
    if (dim == 0) {
        throw std::invalid_argument("graph dimension must be positive");
    }
    if (max_level < 0) {
        throw std::invalid_argument("max_level must be non-negative");
    }
    if (vectors.size() != node_count * dim) {
        throw std::invalid_argument("vectors must be row-major node_count * dim floats");
    }
    if (node_levels.size() != node_count) {
        throw std::invalid_argument("node_levels size must match node_count");
    }
    if (offsets.size() != static_cast<std::size_t>(max_level + 1) * level_stride()) {
        throw std::invalid_argument("offsets size must be (max_level + 1) * (node_count + 1)");
    }
    if (!offsets.empty() && offsets.back() != neighbors.size()) {
        throw std::invalid_argument("final CSR offset must equal neighbors size");
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            throw std::invalid_argument("CSR offsets must be monotonic");
        }
    }
    for (NodeId node : neighbors) {
        if (node >= node_count) {
            throw std::invalid_argument("neighbor id out of range");
        }
    }
}

float l2_squared(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

DescentResult greedy_upper_descent(
    const UpperLayerGraph& graph,
    const float* query,
    NodeId seed,
    int start_level) {
    graph.validate();
    ensure_query(query);
    if (seed >= graph.node_count) {
        throw std::out_of_range("seed id out of range");
    }

    DescentResult result;
    result.seed = seed;
    result.entry_point = seed;
    result.start_level = std::min({start_level, graph.max_level, graph.node_levels[seed]});
    result.entry_distance = l2_squared(
        query,
        graph.vectors.data() + static_cast<std::size_t>(seed) * graph.dim,
        graph.dim);
    result.distance_evaluations = 1;

    NodeId current = seed;
    float current_distance = result.entry_distance;

    for (int level = result.start_level; level > 0; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            ++result.hops;

            const auto [begin, end] = graph.neighbor_range(level, current);
            NodeId best = current;
            float best_distance = current_distance;

            for (const NodeId* it = begin; it != end; ++it) {
                const NodeId candidate = *it;
                const float distance = l2_squared(
                    query,
                    graph.vectors.data() + static_cast<std::size_t>(candidate) * graph.dim,
                    graph.dim);
                ++result.distance_evaluations;
                if (distance < best_distance) {
                    best_distance = distance;
                    best = candidate;
                }
            }

            if (best != current) {
                current = best;
                current_distance = best_distance;
                changed = true;
            }
        }
    }

    result.entry_point = current;
    result.entry_distance = current_distance;
    return result;
}

std::vector<DescentResult> cpu_kway_upper_descent(
    const UpperLayerGraph& graph,
    const float* query,
    const std::vector<NodeId>& seeds,
    std::size_t k) {
    if (k > seeds.size()) {
        throw std::invalid_argument("k exceeds number of seeds");
    }

    std::vector<DescentResult> results;
    results.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        results.push_back(greedy_upper_descent(graph, query, seeds[i], graph.max_level));
    }
    return results;
}

std::vector<SearchResult> search_layer0(
    const UpperLayerGraph& graph,
    const float* query,
    const std::vector<NodeId>& entry_points,
    std::size_t ef,
    std::size_t top_k) {
    graph.validate();
    ensure_query(query);
    if (entry_points.empty()) {
        throw std::invalid_argument("at least one entry point is required");
    }
    ef = std::max(ef, top_k);
    if (ef == 0) {
        throw std::invalid_argument("ef and top_k cannot both be zero");
    }

    std::priority_queue<Candidate, std::vector<Candidate>, MinCandidate> candidate_set;
    std::priority_queue<Candidate, std::vector<Candidate>, MaxCandidate> top_candidates;
    std::vector<unsigned char> visited(graph.node_count, 0);

    auto lower_bound = std::numeric_limits<float>::infinity();
    auto add_top_candidate = [&](NodeId node, float distance) {
        top_candidates.push({distance, node});
        if (top_candidates.size() > ef) {
            top_candidates.pop();
        }
        lower_bound = top_candidates.empty() ? std::numeric_limits<float>::infinity()
                                             : top_candidates.top().distance;
    };

    for (NodeId ep : entry_points) {
        if (ep >= graph.node_count || visited[ep]) continue;
        visited[ep] = 1;
        const float distance = l2_squared(
            query,
            graph.vectors.data() + static_cast<std::size_t>(ep) * graph.dim,
            graph.dim);
        add_top_candidate(ep, distance);
        candidate_set.push({distance, ep});
    }

    while (!candidate_set.empty()) {
        const Candidate current = candidate_set.top();
        if (top_candidates.size() == ef && current.distance > lower_bound) {
            break;
        }
        candidate_set.pop();

        const auto [begin, end] = graph.neighbor_range(0, current.node);
        for (const NodeId* it = begin; it != end; ++it) {
            const NodeId candidate = *it;
            if (visited[candidate]) continue;
            visited[candidate] = 1;

            const float distance = l2_squared(
                query,
                graph.vectors.data() + static_cast<std::size_t>(candidate) * graph.dim,
                graph.dim);
            if (top_candidates.size() < ef || distance < lower_bound) {
                add_top_candidate(candidate, distance);
                candidate_set.push({distance, candidate});
            }
        }
    }

    std::vector<SearchResult> results;
    results.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        const Candidate candidate = top_candidates.top();
        top_candidates.pop();
        results.push_back({candidate.node, candidate.distance});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& lhs, const SearchResult& rhs) {
        if (lhs.distance == rhs.distance) return lhs.node < rhs.node;
        return lhs.distance < rhs.distance;
    });

    if (results.size() > top_k) {
        results.resize(top_k);
    }
    return results;
}

std::vector<NodeId> top_level_seeds(const UpperLayerGraph& graph, std::size_t max_count) {
    graph.validate();
    int seed_level = graph.max_level;
    while (seed_level > 0) {
        std::size_t count = 0;
        for (int level : graph.node_levels) {
            if (level >= seed_level) ++count;
        }
        if (count > 0) break;
        --seed_level;
    }

    std::vector<NodeId> seeds;
    for (NodeId node = 0; node < graph.node_count && seeds.size() < max_count; ++node) {
        if (graph.node_levels[node] >= seed_level) {
            seeds.push_back(node);
        }
    }
    return seeds;
}

std::vector<NodeId> entry_points_from_descent(const std::vector<DescentResult>& descents) {
    std::vector<NodeId> entries;
    entries.reserve(descents.size());
    std::unordered_set<NodeId> seen;
    for (const DescentResult& descent : descents) {
        if (seen.insert(descent.entry_point).second) {
            entries.push_back(descent.entry_point);
        }
    }
    return entries;
}

}  // namespace hnsw_week2

#include "hnsw_week2/gpu_upper_descent.hpp"
#include "hnsw_week2/upper_descent.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using hnsw_week2::NodeId;
using hnsw_week2::UpperLayerGraph;

std::vector<int> parse_csv_ints(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) values.push_back(std::stoi(item));
    }
    return values;
}

std::unordered_map<std::string, std::string> read_meta(const std::string& dir) {
    std::ifstream in(dir + "/meta.txt");
    if (!in) throw std::runtime_error("failed to open meta.txt");

    std::unordered_map<std::string, std::string> meta;
    std::string key;
    std::string value;
    while (in >> key >> value) {
        meta[key] = value;
    }
    return meta;
}

template <typename T>
std::vector<T> read_binary(const std::string& path, std::size_t count) {
    std::vector<T> values(count);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open " + path);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) throw std::runtime_error("failed to read " + path);
    return values;
}

template <typename T>
void write_binary(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open " + path);
    out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) throw std::runtime_error("failed to write " + path);
}

std::vector<NodeId> top_layer_seeds(const UpperLayerGraph& graph, int top_layer_window, std::size_t k) {
    if (top_layer_window <= 0) {
        throw std::invalid_argument("top_layer_window must be positive");
    }

    const int min_node_level = std::max(1, graph.max_level + 1 - top_layer_window);
    std::vector<NodeId> candidates;
    for (NodeId node = 0; node < graph.node_count; ++node) {
        if (graph.node_levels[node] >= min_node_level) {
            candidates.push_back(node);
        }
    }
    if (candidates.empty()) {
        for (NodeId node = 0; node < graph.node_count; ++node) {
            if (graph.node_levels[node] >= 1) candidates.push_back(node);
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("no upper-layer candidates found");
    }

    std::sort(candidates.begin(), candidates.end(), [&](NodeId lhs, NodeId rhs) {
        if (graph.node_levels[lhs] == graph.node_levels[rhs]) return lhs < rhs;
        return graph.node_levels[lhs] > graph.node_levels[rhs];
    });

    std::vector<NodeId> seeds(k);
    for (std::size_t i = 0; i < k; ++i) {
        seeds[i] = candidates[i % candidates.size()];
    }
    std::cout << "top_layer_window=" << top_layer_window
              << " min_node_level=" << min_node_level
              << " candidate_count=" << candidates.size() << '\n';
    return seeds;
}

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " <export_dir> [entry_count_csv=64,260] [upper_window=3] [repeats=3] [output_dir]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    const std::string dir = argv[1];
    const std::vector<int> entry_counts = parse_csv_ints(argc >= 3 ? argv[2] : "64,260");
    const int top_layer_window = argc >= 4 ? std::stoi(argv[3]) : 3;
    const int repeats = argc >= 5 ? std::stoi(argv[4]) : 3;
    const std::string output_dir = argc >= 6 ? argv[5] : "";

    if (!hnsw_week2::cuda_available()) {
        std::cerr << "CUDA is not available\n";
        return 1;
    }

    const auto meta = read_meta(dir);
    const std::string metric = meta.count("metric") == 0 ? "l2" : meta.at("metric");
    UpperLayerGraph graph;
    graph.node_count = static_cast<std::size_t>(std::stoull(meta.at("node_count")));
    graph.dim = static_cast<std::size_t>(std::stoull(meta.at("dim")));
    graph.max_level = std::stoi(meta.at("max_level"));
    const std::size_t query_count = static_cast<std::size_t>(std::stoull(meta.at("query_count")));

    graph.vectors = read_binary<float>(dir + "/vectors.f32", graph.node_count * graph.dim);
    graph.node_levels = read_binary<int>(dir + "/node_levels.i32", graph.node_count);
    graph.offsets = read_binary<std::uint32_t>(
        dir + "/offsets.u32",
        static_cast<std::size_t>(graph.max_level + 1) * (graph.node_count + 1));
    const std::size_t neighbor_count = graph.offsets.back();
    graph.neighbors = read_binary<NodeId>(dir + "/neighbors.u32", neighbor_count);
    std::vector<float> queries = read_binary<float>(dir + "/queries.f32", query_count * graph.dim);
    graph.validate();

    std::cout << "loaded graph nodes=" << graph.node_count
              << " dim=" << graph.dim
              << " max_level=" << graph.max_level
              << " neighbors=" << graph.neighbors.size()
              << " queries=" << query_count
              << " metric=" << metric << '\n';

    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }

    hnsw_week2::GpuUpperDescentIndex gpu_index(graph);
    std::cout << "graph_upload_ms=" << gpu_index.graph_upload_ms() << '\n';

    for (int entry_count_value : entry_counts) {
        if (entry_count_value <= 0) throw std::invalid_argument("entry_count must be positive");
        const std::size_t entry_count = static_cast<std::size_t>(entry_count_value);
        const std::vector<NodeId> seed_row = top_layer_seeds(graph, top_layer_window, entry_count);
        std::vector<NodeId> seeds(query_count * entry_count);
        for (std::size_t q = 0; q < query_count; ++q) {
            std::copy(seed_row.begin(), seed_row.end(), seeds.begin() + q * entry_count);
        }

        for (int repeat = 0; repeat < repeats; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            hnsw_week2::GpuUpperDescentTiming timing;
            const std::vector<hnsw_week2::DescentResult> descents =
                gpu_index.search(queries, query_count, seeds, entry_count, &timing);
            const auto stop = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(stop - start).count();

            double best_distance_sum = 0.0;
            std::vector<int> best_entries(query_count);
            std::vector<float> best_scores(query_count);
            for (std::size_t q = 0; q < query_count; ++q) {
                float best = std::numeric_limits<float>::infinity();
                NodeId best_entry = 0;
                for (std::size_t i = 0; i < entry_count; ++i) {
                    const auto& descent = descents[q * entry_count + i];
                    if (descent.entry_distance < best) {
                        best = descent.entry_distance;
                        best_entry = descent.entry_point;
                    }
                }
                best_distance_sum += best;
                best_entries[q] = static_cast<int>(best_entry);
                best_scores[q] = metric == "ip" ? 1.0f - 0.5f * best : best;
            }

            std::cout << "entry_count=" << entry_count
                      << " repeat=" << repeat
                      << " resident_total_ms=" << ms
                      << " input_upload_ms=" << timing.input_upload_ms
                      << " kernel_ms=" << timing.kernel_ms
                      << " result_download_ms=" << timing.result_download_ms
                      << " per_query_ms=" << (ms / static_cast<double>(query_count))
                      << " avg_best_l2=" << (best_distance_sum / static_cast<double>(query_count))
                      << '\n';

            if (repeat == 0 && !output_dir.empty()) {
                const std::string prefix =
                    output_dir + "/gpu_best_k" + std::to_string(entry_count) + "_w" +
                    std::to_string(top_layer_window);
                write_binary(prefix + "_nearest.i32", best_entries);
                write_binary(prefix + "_nearest_d.f32", best_scores);
                std::ofstream meta_out(prefix + "_meta.txt");
                meta_out << "query_count " << query_count << '\n';
                meta_out << "entry_count " << entry_count << '\n';
                meta_out << "upper_window " << top_layer_window << '\n';
                meta_out << "metric " << metric << '\n';
                meta_out << "resident_total_ms " << ms << '\n';
                meta_out << "input_upload_ms " << timing.input_upload_ms << '\n';
                meta_out << "kernel_ms " << timing.kernel_ms << '\n';
                meta_out << "result_download_ms " << timing.result_download_ms << '\n';
            }
        }
    }

    return 0;
}

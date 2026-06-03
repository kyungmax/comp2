#include "hnsw_week2/gpu_upper_descent.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hnsw_week2 {

namespace {

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
 public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) {
        reset(count);
    }
    ~DeviceBuffer() {
        cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const {
        return ptr_;
    }

    void reset(std::size_t count) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)), "cudaMalloc");
        }
    }

    void copy_from_host(const T* src, std::size_t count) {
        if (count == 0) return;
        check_cuda(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }

    void copy_to_host(T* dst, std::size_t count) const {
        if (count == 0) return;
        check_cuda(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }

 private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

__device__ float warp_reduce_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__device__ float warp_l2_squared(
    const float* query,
    const float* vectors,
    std::uint32_t node,
    int dim,
    int lane) {
    float sum = 0.0f;
    const float* vector = vectors + static_cast<std::size_t>(node) * dim;
    for (int d = lane; d < dim; d += 32) {
        const float diff = query[d] - vector[d];
        sum += diff * diff;
    }
    sum = warp_reduce_sum(sum);
    return __shfl_sync(0xffffffffu, sum, 0);
}

__global__ void kway_upper_descent_kernel(
    const float* vectors,
    const float* queries,
    const int* node_levels,
    const std::uint32_t* offsets,
    const std::uint32_t* neighbors,
    int node_count,
    int dim,
    int max_level,
    const std::uint32_t* seeds,
    int k,
    int warps_per_block,
    std::uint32_t* out_seed,
    std::uint32_t* out_entry,
    float* out_distance,
    unsigned long long* out_evaluations,
    unsigned long long* out_hops) {
    constexpr unsigned mask = 0xffffffffu;
    const int query_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int seed_id = blockIdx.y * warps_per_block + warp_id;
    if (warp_id >= warps_per_block || seed_id >= k) return;

    const int out_index = query_id * k + seed_id;
    const std::uint32_t seed = seeds[out_index];
    const float* query = queries + static_cast<std::size_t>(query_id) * dim;

    std::uint32_t current = seed;
    int start_level = max_level;
    if (current < static_cast<std::uint32_t>(node_count)) {
        start_level = start_level < node_levels[current] ? start_level : node_levels[current];
    }

    float current_distance = warp_l2_squared(query, vectors, current, dim, lane);
    unsigned long long evaluations = 1;
    unsigned long long hops = 0;

    for (int level = start_level; level > 0; --level) {
        bool changed = true;
        while (__shfl_sync(mask, changed ? 1 : 0, 0)) {
            changed = false;
            if (lane == 0) ++hops;

            std::uint32_t next = current;
            float next_distance = current_distance;
            const std::size_t base = static_cast<std::size_t>(level) * (node_count + 1);
            const std::uint32_t begin = offsets[base + current];
            const std::uint32_t end = offsets[base + current + 1];

            for (std::uint32_t pos = begin; pos < end; ++pos) {
                const std::uint32_t candidate = neighbors[pos];
                const float distance = warp_l2_squared(query, vectors, candidate, dim, lane);
                if (lane == 0) {
                    ++evaluations;
                    if (distance < next_distance) {
                        next_distance = distance;
                        next = candidate;
                    }
                }
            }

            next = __shfl_sync(mask, next, 0);
            next_distance = __shfl_sync(mask, next_distance, 0);
            const int changed_int = (lane == 0 && next != current) ? 1 : 0;
            changed = __shfl_sync(mask, changed_int, 0) != 0;
            current = next;
            current_distance = next_distance;
        }
    }

    if (lane == 0) {
        out_seed[out_index] = seed;
        out_entry[out_index] = current;
        out_distance[out_index] = current_distance;
        out_evaluations[out_index] = evaluations;
        out_hops[out_index] = hops;
    }
}

}  // namespace

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<DescentResult> gpu_kway_upper_descent(
    const UpperLayerGraph& graph,
    const std::vector<float>& queries,
    std::size_t query_count,
    const std::vector<NodeId>& seeds,
    std::size_t k) {
    graph.validate();
    if (k == 0 || query_count == 0) {
        return {};
    }
    if (queries.size() != query_count * graph.dim) {
        throw std::invalid_argument("queries must be query_count * graph.dim floats");
    }
    if (seeds.size() != query_count * k) {
        throw std::invalid_argument("seeds must contain query_count * k entries");
    }

    const std::size_t output_count = query_count * k;

    DeviceBuffer<float> d_vectors(graph.vectors.size());
    DeviceBuffer<float> d_queries(queries.size());
    DeviceBuffer<int> d_node_levels(graph.node_levels.size());
    DeviceBuffer<std::uint32_t> d_offsets(graph.offsets.size());
    DeviceBuffer<std::uint32_t> d_neighbors(graph.neighbors.size());
    DeviceBuffer<std::uint32_t> d_seeds(seeds.size());
    DeviceBuffer<std::uint32_t> d_out_seed(output_count);
    DeviceBuffer<std::uint32_t> d_out_entry(output_count);
    DeviceBuffer<float> d_out_distance(output_count);
    DeviceBuffer<unsigned long long> d_out_evaluations(output_count);
    DeviceBuffer<unsigned long long> d_out_hops(output_count);

    d_vectors.copy_from_host(graph.vectors.data(), graph.vectors.size());
    d_queries.copy_from_host(queries.data(), queries.size());
    d_node_levels.copy_from_host(graph.node_levels.data(), graph.node_levels.size());
    d_offsets.copy_from_host(graph.offsets.data(), graph.offsets.size());
    d_neighbors.copy_from_host(graph.neighbors.data(), graph.neighbors.size());
    d_seeds.copy_from_host(seeds.data(), seeds.size());

    const int warps_per_block = static_cast<int>(std::min<std::size_t>(8, k));
    const dim3 blocks(
        static_cast<unsigned int>(query_count),
        static_cast<unsigned int>((k + warps_per_block - 1) / warps_per_block));
    const dim3 threads(static_cast<unsigned int>(warps_per_block * 32));
    kway_upper_descent_kernel<<<blocks, threads>>>(
        d_vectors.get(),
        d_queries.get(),
        d_node_levels.get(),
        d_offsets.get(),
        d_neighbors.get(),
        static_cast<int>(graph.node_count),
        static_cast<int>(graph.dim),
        graph.max_level,
        d_seeds.get(),
        static_cast<int>(k),
        warps_per_block,
        d_out_seed.get(),
        d_out_entry.get(),
        d_out_distance.get(),
        d_out_evaluations.get(),
        d_out_hops.get());
    check_cuda(cudaGetLastError(), "launch kway_upper_descent_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<std::uint32_t> out_seed(output_count);
    std::vector<std::uint32_t> out_entry(output_count);
    std::vector<float> out_distance(output_count);
    std::vector<unsigned long long> out_evaluations(output_count);
    std::vector<unsigned long long> out_hops(output_count);

    d_out_seed.copy_to_host(out_seed.data(), output_count);
    d_out_entry.copy_to_host(out_entry.data(), output_count);
    d_out_distance.copy_to_host(out_distance.data(), output_count);
    d_out_evaluations.copy_to_host(out_evaluations.data(), output_count);
    d_out_hops.copy_to_host(out_hops.data(), output_count);

    std::vector<DescentResult> results;
    results.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        DescentResult result;
        result.seed = out_seed[i];
        result.entry_point = out_entry[i];
        result.entry_distance = out_distance[i];
        result.start_level = std::min(graph.max_level, graph.node_levels[out_seed[i]]);
        result.distance_evaluations = static_cast<std::size_t>(out_evaluations[i]);
        result.hops = static_cast<std::size_t>(out_hops[i]);
        results.push_back(result);
    }
    return results;
}

struct GpuUpperDescentIndex::Impl {
    std::size_t node_count = 0;
    std::size_t dim = 0;
    int max_level = 0;
    std::vector<int> node_levels;
    double graph_upload_ms = 0.0;

    DeviceBuffer<float> d_vectors;
    DeviceBuffer<int> d_node_levels;
    DeviceBuffer<std::uint32_t> d_offsets;
    DeviceBuffer<std::uint32_t> d_neighbors;
};

GpuUpperDescentIndex::GpuUpperDescentIndex(const UpperLayerGraph& graph)
    : impl_(std::make_unique<Impl>()) {
    graph.validate();
    impl_->node_count = graph.node_count;
    impl_->dim = graph.dim;
    impl_->max_level = graph.max_level;
    impl_->node_levels = graph.node_levels;

    const auto start = std::chrono::steady_clock::now();
    impl_->d_vectors.reset(graph.vectors.size());
    impl_->d_node_levels.reset(graph.node_levels.size());
    impl_->d_offsets.reset(graph.offsets.size());
    impl_->d_neighbors.reset(graph.neighbors.size());
    impl_->d_vectors.copy_from_host(graph.vectors.data(), graph.vectors.size());
    impl_->d_node_levels.copy_from_host(graph.node_levels.data(), graph.node_levels.size());
    impl_->d_offsets.copy_from_host(graph.offsets.data(), graph.offsets.size());
    impl_->d_neighbors.copy_from_host(graph.neighbors.data(), graph.neighbors.size());
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize graph upload");
    const auto stop = std::chrono::steady_clock::now();
    impl_->graph_upload_ms = std::chrono::duration<double, std::milli>(stop - start).count();
}

GpuUpperDescentIndex::~GpuUpperDescentIndex() = default;

double GpuUpperDescentIndex::graph_upload_ms() const {
    return impl_->graph_upload_ms;
}

std::vector<DescentResult> GpuUpperDescentIndex::search(
    const std::vector<float>& queries,
    std::size_t query_count,
    const std::vector<NodeId>& seeds,
    std::size_t k,
    GpuUpperDescentTiming* timing) const {
    if (k == 0 || query_count == 0) {
        return {};
    }
    if (queries.size() != query_count * impl_->dim) {
        throw std::invalid_argument("queries must be query_count * dim floats");
    }
    if (seeds.size() != query_count * k) {
        throw std::invalid_argument("seeds must contain query_count * k entries");
    }

    const std::size_t output_count = query_count * k;
    DeviceBuffer<float> d_queries(queries.size());
    DeviceBuffer<std::uint32_t> d_seeds(seeds.size());
    DeviceBuffer<std::uint32_t> d_out_seed(output_count);
    DeviceBuffer<std::uint32_t> d_out_entry(output_count);
    DeviceBuffer<float> d_out_distance(output_count);
    DeviceBuffer<unsigned long long> d_out_evaluations(output_count);
    DeviceBuffer<unsigned long long> d_out_hops(output_count);

    GpuUpperDescentTiming local_timing;
    const auto upload_start = std::chrono::steady_clock::now();
    d_queries.copy_from_host(queries.data(), queries.size());
    d_seeds.copy_from_host(seeds.data(), seeds.size());
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize input upload");
    const auto upload_stop = std::chrono::steady_clock::now();
    local_timing.input_upload_ms =
        std::chrono::duration<double, std::milli>(upload_stop - upload_start).count();

    const int warps_per_block = static_cast<int>(std::min<std::size_t>(8, k));
    const dim3 blocks(
        static_cast<unsigned int>(query_count),
        static_cast<unsigned int>((k + warps_per_block - 1) / warps_per_block));
    const dim3 threads(static_cast<unsigned int>(warps_per_block * 32));

    const auto kernel_start = std::chrono::steady_clock::now();
    kway_upper_descent_kernel<<<blocks, threads>>>(
        impl_->d_vectors.get(),
        d_queries.get(),
        impl_->d_node_levels.get(),
        impl_->d_offsets.get(),
        impl_->d_neighbors.get(),
        static_cast<int>(impl_->node_count),
        static_cast<int>(impl_->dim),
        impl_->max_level,
        d_seeds.get(),
        static_cast<int>(k),
        warps_per_block,
        d_out_seed.get(),
        d_out_entry.get(),
        d_out_distance.get(),
        d_out_evaluations.get(),
        d_out_hops.get());
    check_cuda(cudaGetLastError(), "launch kway_upper_descent_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize resident search");
    const auto kernel_stop = std::chrono::steady_clock::now();
    local_timing.kernel_ms =
        std::chrono::duration<double, std::milli>(kernel_stop - kernel_start).count();

    std::vector<std::uint32_t> out_seed(output_count);
    std::vector<std::uint32_t> out_entry(output_count);
    std::vector<float> out_distance(output_count);
    std::vector<unsigned long long> out_evaluations(output_count);
    std::vector<unsigned long long> out_hops(output_count);

    const auto download_start = std::chrono::steady_clock::now();
    d_out_seed.copy_to_host(out_seed.data(), output_count);
    d_out_entry.copy_to_host(out_entry.data(), output_count);
    d_out_distance.copy_to_host(out_distance.data(), output_count);
    d_out_evaluations.copy_to_host(out_evaluations.data(), output_count);
    d_out_hops.copy_to_host(out_hops.data(), output_count);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize result download");
    const auto download_stop = std::chrono::steady_clock::now();
    local_timing.result_download_ms =
        std::chrono::duration<double, std::milli>(download_stop - download_start).count();
    if (timing != nullptr) {
        *timing = local_timing;
    }

    std::vector<DescentResult> results;
    results.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        DescentResult result;
        result.seed = out_seed[i];
        result.entry_point = out_entry[i];
        result.entry_distance = out_distance[i];
        result.start_level = std::min(impl_->max_level, impl_->node_levels[out_seed[i]]);
        result.distance_evaluations = static_cast<std::size_t>(out_evaluations[i]);
        result.hops = static_cast<std::size_t>(out_hops[i]);
        results.push_back(result);
    }
    return results;
}

}  // namespace hnsw_week2

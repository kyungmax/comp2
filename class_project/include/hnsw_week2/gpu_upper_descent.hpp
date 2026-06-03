#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "hnsw_week2/upper_descent.hpp"

namespace hnsw_week2 {

bool cuda_available();

std::vector<DescentResult> gpu_kway_upper_descent(
    const UpperLayerGraph& graph,
    const std::vector<float>& queries,
    std::size_t query_count,
    const std::vector<NodeId>& seeds,
    std::size_t k);

struct GpuUpperDescentTiming {
    double input_upload_ms = 0.0;
    double kernel_ms = 0.0;
    double result_download_ms = 0.0;
};

class GpuUpperDescentIndex {
 public:
    explicit GpuUpperDescentIndex(const UpperLayerGraph& graph);
    ~GpuUpperDescentIndex();

    GpuUpperDescentIndex(const GpuUpperDescentIndex&) = delete;
    GpuUpperDescentIndex& operator=(const GpuUpperDescentIndex&) = delete;

    double graph_upload_ms() const;

    std::vector<DescentResult> search(
        const std::vector<float>& queries,
        std::size_t query_count,
        const std::vector<NodeId>& seeds,
        std::size_t k,
        GpuUpperDescentTiming* timing = nullptr) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hnsw_week2

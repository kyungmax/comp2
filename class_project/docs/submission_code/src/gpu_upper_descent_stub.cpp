#include "hnsw_week2/gpu_upper_descent.hpp"

#include <stdexcept>

namespace hnsw_week2 {

bool cuda_available() {
    return false;
}

std::vector<DescentResult> gpu_kway_upper_descent(
    const UpperLayerGraph&,
    const std::vector<float>&,
    std::size_t,
    const std::vector<NodeId>&,
    std::size_t) {
    throw std::runtime_error("CUDA support was not built; configure with nvcc to enable GPU descent");
}

struct GpuUpperDescentIndex::Impl {};

GpuUpperDescentIndex::GpuUpperDescentIndex(const UpperLayerGraph&) {
    throw std::runtime_error("CUDA support was not built; configure with nvcc to enable GPU descent");
}

GpuUpperDescentIndex::~GpuUpperDescentIndex() = default;

double GpuUpperDescentIndex::graph_upload_ms() const {
    return 0.0;
}

std::vector<DescentResult> GpuUpperDescentIndex::search(
    const std::vector<float>&,
    std::size_t,
    const std::vector<NodeId>&,
    std::size_t,
    GpuUpperDescentTiming*) const {
    throw std::runtime_error("CUDA support was not built; configure with nvcc to enable GPU descent");
}

}  // namespace hnsw_week2

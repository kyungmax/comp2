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

}  // namespace hnsw_week2

#pragma once

#include <cstddef>
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

}  // namespace hnsw_week2

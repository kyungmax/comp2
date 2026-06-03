cmake_minimum_required(VERSION 3.24)

project(hnsw_week2 LANGUAGES CXX)

option(HNSW_WEEK2_ENABLE_CUDA "Build the CUDA upper-layer descent implementation when nvcc is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hnsw_week2_cpu
    src/upper_descent.cpp
)
target_include_directories(hnsw_week2_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(CheckLanguage)
check_language(CUDA)

if(HNSW_WEEK2_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    add_library(hnsw_week2_gpu
        src/gpu_upper_descent.cu
    )
    set_target_properties(hnsw_week2_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_definitions(hnsw_week2_gpu PUBLIC HNSW_WEEK2_HAS_CUDA=1)
else()
    add_library(hnsw_week2_gpu
        src/gpu_upper_descent_stub.cpp
    )
endif()

target_link_libraries(hnsw_week2_gpu PUBLIC hnsw_week2_cpu)
target_include_directories(hnsw_week2_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(week2_parity
    tests/week2_parity.cpp
)
target_link_libraries(week2_parity PRIVATE hnsw_week2_gpu)

add_executable(bench_gpu_upper_descent
    src/bench_gpu_upper_descent.cpp
)
target_link_libraries(bench_gpu_upper_descent PRIVATE hnsw_week2_gpu)

enable_testing()
add_test(NAME week2_parity COMMAND week2_parity)

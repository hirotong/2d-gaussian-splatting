cmake_minimum_required(VERSION 3.20)

project(BVH LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(BVHTracer
    src/trace.cu
    src/construct.cu
    src/bvh.cu
)


set_target_properties(BVHTracer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(BVHTracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

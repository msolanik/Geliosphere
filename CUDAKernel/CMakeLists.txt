cmake_minimum_required(VERSION 3.10)

project(Cuda-kernel CUDA CXX)

include_directories(include)

find_package(CUDA REQUIRED)

file(GLOB SOURCES "src/*.cu" "src/*.cpp")

set(Cuda-kernel ${SOURCES}) 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -Xcompiler -O3 --use_fast_math --resource-usage")
set(cpulib "cpu")
set(gpulib "gpu")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

add_library(Cuda-kernel SHARED ${SOURCES})
add_library(Cuda-kernel::Cuda-kernel ALIAS ${PROJECT_NAME})

set_target_properties(Cuda-kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(default_build_type "Release")

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    Constants::Basic
    utils::utils
)
# Set the project name
project (Algorithm CUDA CXX)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
set(Algorithm ${SOURCES})  
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

find_package(CUDA REQUIRED)

add_library(Algorithm SHARED ${SOURCES}) 
add_library(Algorithm::Algorithm ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    Constants::Basic
    cpu_implementations::cpu_implementations
    Cuda-kernel::Cuda-kernel
    ${CUDA_LIBRARIES}
)
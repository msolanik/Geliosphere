# Set the project name
project (Algorithm)

include_directories(include)

include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.9.2
)
FetchContent_MakeAvailable(spdlog)

file(GLOB SOURCES "src/*.cpp")
set(Algorithm ${SOURCES})  
# set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O3")

add_library(Algorithm SHARED ${SOURCES}) 
add_library(Algorithm::Algorithm ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

if(CPU_VERSION_ONLY)
    target_link_libraries(${PROJECT_NAME}
    Constants::Basic
    cpu_implementations::cpu_implementations
    spdlog::spdlog_header_only
    )
else()
    target_link_libraries(${PROJECT_NAME}
    Constants::Basic
    cpu_implementations::cpu_implementations
    Cuda-kernel::Cuda-kernel
    spdlog::spdlog_header_only
    )
endif()

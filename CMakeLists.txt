cmake_minimum_required(VERSION 3.14)
set( CMAKE_VERBOSE_MAKEFILE on )
# set the project name
project(Geliosphere)
option(CPU_VERSION_ONLY "Build only CPU version" OFF)

if(CPU_VERSION_ONLY)
  add_definitions(-DGPU_ENABLED=0)
else()
  add_definitions(-DGPU_ENABLED=1)
endif()


include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.9.2
)
FetchContent_MakeAvailable(spdlog)

add_subdirectory (Algorithm)
add_subdirectory (Constants)
add_subdirectory (CpuImplementations)
if(CPU_VERSION_ONLY MATCHES OFF)
add_subdirectory (CUDAKernel)
endif()
add_subdirectory (Factory)
add_subdirectory (Utils)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

configure_file(SolarProp_K0_phi_table.csv SolarProp_K0_phi_table.csv COPYONLY)
configure_file(Geliosphere_K0_phi_table.csv Geliosphere_K0_phi_table.csv COPYONLY)
configure_file(K0_phi_table.csv K0_phi_table.csv COPYONLY)
configure_file(Settings.toml Settings.toml COPYONLY)

# add the executable
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    Constants::Basic
    Algorithm::Algorithm
    factory::factory
    spdlog::spdlog_header_only
)
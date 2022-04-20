# Set the project name
project (cpu_implementations)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
set(cpu_implementations ${SOURCES})  
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

add_library(cpu_implementations SHARED ${SOURCES}) 
add_library(cpu_implementations::cpu_implementations ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    Constants::Basic
    utils::utils
    Threads::Threads
)
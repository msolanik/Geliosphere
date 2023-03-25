# Set the project name
project (factory)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
set(factory ${SOURCES})  
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

add_library(factory SHARED ${SOURCES}) 
add_library(factory::factory ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    Input::Basic
    Algorithm::Algorithm
)
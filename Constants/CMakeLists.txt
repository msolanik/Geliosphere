# Set the project name
project (Constants)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
set(Constants ${SOURCES})  
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

add_library(Constants SHARED ${SOURCES}) 
add_library(Constants::Basic ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    cli_dependencies
)
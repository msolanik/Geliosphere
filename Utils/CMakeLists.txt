# Set the project name
project (utils)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
set(utils ${SOURCES})  
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O3")

add_library(utils SHARED ${SOURCES}) 
add_library(utils::utils ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    Input::Basic
)
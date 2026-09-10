
cmake_minimum_required(VERSION 3.10)

project(StandaloneORB VERSION 1.0 LANGUAGES CXX)

# Require C++14 standard or higher
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add maximum optimization flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
endif()

# Core algorithm sources, shared by the CLI executable and the test suite
set(LIB_SOURCES
    orb_core.cpp
    fast_detector.cpp
    image_utils.cpp
    orb.cpp
)

# Define the header files (useful to make them appear in IDEs)
set(HEADERS
    orb_core.hpp
    fast_detector.hpp
    image_utils.hpp
    orb_pattern.hpp
    orb.hpp
    stb_image.h
)

add_library(orb_lib STATIC ${LIB_SOURCES} ${HEADERS})
target_include_directories(orb_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create the main executable
add_executable(orb_test main.cpp)
target_link_libraries(orb_test PRIVATE orb_lib)

enable_testing()
add_subdirectory(tests)

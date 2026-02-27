
cmake_minimum_required(VERSION 3.10)

project(StandaloneORB VERSION 1.0 LANGUAGES CXX)

# Require C++14 standard or higher
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add maximum optimization flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
endif()

# Define the source files created so far
set(SOURCES
    orb_core.cpp
    fast_detector.cpp
    image_utils.cpp
    orb.cpp
    main.cpp
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

# Create the main executable
add_executable(orb_test ${SOURCES} ${HEADERS})

# Specify the include directories
target_include_directories(orb_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
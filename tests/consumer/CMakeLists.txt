# Minimal external project consuming an installed StandaloneORB package.
# Not part of the main build: CI installs the library to a prefix and
# configures this directory with CMAKE_PREFIX_PATH pointing at it (#28).
cmake_minimum_required(VERSION 3.11)
project(OrbConsumer LANGUAGES CXX)

find_package(StandaloneORB 0.2 REQUIRED)

add_executable(orb_consumer main.cpp)
target_link_libraries(orb_consumer PRIVATE StandaloneORB::orb)

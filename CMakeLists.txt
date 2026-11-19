
cmake_minimum_required(VERSION 3.11)

# The version follows the release tags (v0.1.3 is the last one; the next
# release bumps this as part of the release PR, see AGENTS.md).
project(StandaloneORB VERSION 0.2.0 LANGUAGES CXX)

# Tools (orb_test, profile_orb) and the test suite are built by default only
# when this is the top-level project; a consumer that add_subdirectory()s the
# repo gets just the library (issue #28).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ORB_IS_TOP_LEVEL ON)
else()
    set(ORB_IS_TOP_LEVEL OFF)
endif()
option(ORB_BUILD_TOOLS "Build orb_test and profile_orb" ${ORB_IS_TOP_LEVEL})
option(ORB_BUILD_TESTS "Build the orb_tests suite" ${ORB_IS_TOP_LEVEL})
option(ORB_INSTALL "Generate install rules and the StandaloneORB CMake package" ${ORB_IS_TOP_LEVEL})

# Require C++14 standard or higher
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add maximum optimization flags for performance (directory scope, so the
# tools and tests get them too).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
endif()

# WebAssembly via Emscripten (`emcmake cmake ...`, issue #27). The toolchain
# file sets EMSCRIPTEN, makes executables .js + .wasm and registers node as
# the CMAKE_CROSSCOMPILING_EMULATOR, so ctest runs the suite under node.
# - -fexceptions: the constructor throws std::invalid_argument and the tests
#   catch it; Emscripten disables exception catching by default.
# - NODERAWFS: executables read the real file system (fixtures, images).
# - ALLOW_MEMORY_GROWTH: the 1637x2048 sample image needs more than the
#   16 MB initial heap. EXIT_RUNTIME: main()'s return value is the exit code.
if(EMSCRIPTEN)
    add_compile_options(-fexceptions)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fexceptions -sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1")
endif()

# Build every target with AddressSanitizer + UndefinedBehaviorSanitizer.
# Off by default; CI runs the test suite with it on (see .github/workflows/ci.yml).
option(ORB_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(ORB_SANITIZE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
        # Linker flag variables rather than add_link_options(), which needs CMake 3.13.
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address,undefined")
    elseif(MSVC)
        # MSVC has ASan only (no UBSan); it is incompatible with /RTC1 in Debug.
        # To run the binaries, the ASan runtime directory must be on PATH, e.g.
        # <VS>/VC/Tools/MSVC/<ver>/bin/Hostx64/x64 (otherwise exit code 0xc0000135).
        add_compile_options(/fsanitize=address)
        string(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    else()
        message(WARNING "ORB_SANITIZE is not supported for ${CMAKE_CXX_COMPILER_ID}; ignoring")
    endif()
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
# Consumers use the namespaced name, whether they add_subdirectory() this repo
# or find_package(StandaloneORB) an installed copy.
add_library(StandaloneORB::orb ALIAS orb_lib)
set_target_properties(orb_lib PROPERTIES EXPORT_NAME orb OUTPUT_NAME orb)
target_compile_features(orb_lib PUBLIC cxx_std_14)
target_include_directories(orb_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/StandaloneORB>)

# -ffp-contract=off: the Gaussian blur reproduces OpenCV's float arithmetic
# bit for bit (issue #39), which needs a*b + c to stay two roundings; GCC
# fuses it into an FMA on targets that have one (e.g. -march=x86-64-v3),
# even across an explicit cast, and that moves rounding-tie pixels by one.
# On the target, not in CMAKE_CXX_FLAGS, so a subproject build keeps it.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(orb_lib PRIVATE -ffp-contract=off)
endif()

if(ORB_BUILD_TOOLS)
    # Create the main executable
    add_executable(orb_test main.cpp)
    target_link_libraries(orb_test PRIVATE StandaloneORB::orb)

    # Per-stage timing diagnostic (pyramid, FAST, Harris, orientation,
    # descriptors, ...). No external deps, built by default like orb_test.
    add_executable(profile_orb tools/profile_orb.cpp)
    target_link_libraries(profile_orb PRIVATE StandaloneORB::orb)
endif()

if(ORB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ORB_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS orb_lib EXPORT StandaloneORBTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    # The public headers; stb_image.h is only used by the tools.
    install(FILES orb.hpp orb_core.hpp fast_detector.hpp image_utils.hpp orb_pattern.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/StandaloneORB)
    install(EXPORT StandaloneORBTargets
        NAMESPACE StandaloneORB::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StandaloneORB)
    configure_package_config_file(cmake/StandaloneORBConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/StandaloneORBConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StandaloneORB)
    # 0.x: output and API change between minor versions (e.g. the constructor
    # started throwing in 0.1.3), so only the same minor is compatible.
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/StandaloneORBConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/StandaloneORBConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/StandaloneORBConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StandaloneORB)
endif()

# Optional diagnostic tool comparing this repo's ORB against OpenCV's cv::ORB.
# Off by default: orb_lib/orb_test/orb_tests never require OpenCV.
# See docs/design/compare-orb-design.md.
option(BUILD_OPENCV_COMPARISON "Build the compare_orb tool (requires OpenCV)" OFF)

if(BUILD_OPENCV_COMPARISON)
    find_package(OpenCV REQUIRED)
    add_executable(compare_orb tools/compare_orb.cpp)
    target_include_directories(compare_orb PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        # Fallback for the official OpenCV Windows package: its per-toolset
        # OpenCVConfig.cmake (under build/x64/vcXX/lib) bakes in absolute
        # include paths from the OpenCV build machine, which don't exist
        # locally. The real headers live at the standard build/include path.
        "${OpenCV_DIR}/../../../include"
    )
    # Prefer the single "world" target when the OpenCV package only ships
    # that (its OpenCV_LIBS can otherwise list per-module libs that don't
    # exist in a world-only build).
    if(TARGET opencv_world)
        target_link_libraries(compare_orb PRIVATE StandaloneORB::orb opencv_world)
    else()
        target_link_libraries(compare_orb PRIVATE StandaloneORB::orb ${OpenCV_LIBS})
    endif()
endif()

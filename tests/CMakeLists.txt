add_executable(orb_tests
    test_main.cpp
    test_image_utils.cpp
    test_fast_detector.cpp
    test_orb_core.cpp
    test_orb.cpp
    test_opencv_ref.cpp
)

target_link_libraries(orb_tests PRIVATE StandaloneORB::orb)

# Where the committed OpenCV ground-truth fixtures live (issue #32). The tests
# read the PGM files from here at run time; the generated header is included.
target_compile_definitions(orb_tests PRIVATE
    ORB_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/opencv"
    ORB_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    # What orbSimdActive() must report for this build (see test_orb_core.cpp).
    ORB_TEST_EXPECT_SIMD=$<BOOL:${ORB_ENABLE_SIMD}>)

add_test(NAME orb_tests COMMAND orb_tests)

# Dumps the fixture input image once (tests/fixtures/opencv/scene-640x480.pgm).
# Developer tool, not run by ctest.
add_executable(dump_scene dump_scene.cpp)
target_link_libraries(dump_scene PRIVATE StandaloneORB::orb)

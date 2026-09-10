add_executable(orb_tests
    test_main.cpp
    test_image_utils.cpp
    test_fast_detector.cpp
    test_orb_core.cpp
    test_orb.cpp
)

target_link_libraries(orb_tests PRIVATE orb_lib)

add_test(NAME orb_tests COMMAND orb_tests)

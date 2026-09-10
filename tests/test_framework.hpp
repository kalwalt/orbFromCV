#ifndef TEST_FRAMEWORK_HPP
#define TEST_FRAMEWORK_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// Minimal header-only test harness. Kept dependency-free to match the
// project's "no external deps beyond stb_image.h" convention.

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

struct TestRegistrar {
    TestRegistrar(const std::string& name, const std::function<void()>& fn) {
        test_registry().push_back({name, fn});
    }
};

inline int& test_failure_count() {
    static int failures = 0;
    return failures;
}

#define TEST_CONCAT_INNER(a, b) a##b
#define TEST_CONCAT(a, b) TEST_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                                      \
    static void TEST_CONCAT(test_fn_, __LINE__)();                           \
    static TestRegistrar TEST_CONCAT(test_reg_, __LINE__)(                   \
        name, TEST_CONCAT(test_fn_, __LINE__));                              \
    static void TEST_CONCAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "    CHECK failed: " #cond << " (" << __FILE__      \
                       << ":" << __LINE__ << ")" << std::endl;               \
            ++test_failure_count();                                          \
        }                                                                    \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

inline bool approxEqual(double a, double b, double eps = 1e-3) {
    return std::fabs(a - b) <= eps;
}

#define CHECK_NEAR(a, b, eps)                                                \
    do {                                                                     \
        if (!approxEqual((a), (b), (eps))) {                                 \
            std::cerr << "    CHECK_NEAR failed: " #a " ~= " #b << " (got "  \
                       << (a) << " vs " << (b) << " at " << __FILE__ << ":"  \
                       << __LINE__ << ")" << std::endl;                      \
            ++test_failure_count();                                         \
        }                                                                    \
    } while (0)

#endif // TEST_FRAMEWORK_HPP

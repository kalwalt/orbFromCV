#include "test_framework.hpp"

int main() {
    int total = 0;
    int failed = 0;

    for (const auto& tc : test_registry()) {
        int before = test_failure_count();
        tc.fn();
        ++total;
        if (test_failure_count() > before) {
            std::cerr << "FAILED: " << tc.name << std::endl;
            ++failed;
        } else {
            std::cout << "PASSED: " << tc.name << std::endl;
        }
    }

    std::cout << (total - failed) << "/" << total << " tests passed."
               << std::endl;
    return failed == 0 ? 0 : 1;
}

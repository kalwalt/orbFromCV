#include "thread_pool.hpp"
#include "test_framework.hpp"
#include <stdexcept>
#include <vector>

TEST_CASE("ThreadPool::parallelFor: covers the range once, in band order, with empty bands skipped") {
    ThreadPool pool(3);
    for (int n : {0, 1, 2, 3, 4, 5, 17, 100}) {
        std::vector<int> hits(n, 0);
        std::vector<int> bandOf(n, -1);
        pool.parallelFor(0, n, [&](int band, int b, int e) {
            CHECK(b < e);
            for (int i = b; i < e; ++i) { ++hits[i]; bandOf[i] = band; }
        });
        for (int i = 0; i < n; ++i) CHECK_EQ(hits[i], 1);
        for (int i = 1; i < n; ++i) CHECK(bandOf[i] >= bandOf[i - 1]);
    }
}

TEST_CASE("ThreadPool::parallelFor: an exception in a band reaches the caller") {
    ThreadPool pool(2);
    bool caught = false;
    try {
        pool.parallelFor(0, 8, [](int, int b, int) { if (b > 0) throw std::runtime_error("band"); });
    } catch (const std::runtime_error&) { caught = true; }
    CHECK(caught);
    int sum = 0;  // the pool is still usable afterwards
    pool.parallelFor(0, 8, [&](int, int b, int e) { sum += e - b; });
    CHECK_EQ(sum, 8);
}

#include "thread_pool.hpp"
#include "test_framework.hpp"
#include <atomic>
#include <stdexcept>
#include <vector>

// A wasm32 build without -pthread has no std::thread to start; ORB clamps
// setNumThreads to 1 there (test_orb.cpp) and these cases are compiled out.
#if !(defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__))

TEST_CASE("ThreadPool::parallelFor: covers the range once, in band order, with empty bands skipped") {
    ThreadPool pool(3);
    for (int n : {0, 1, 2, 3, 4, 5, 17, 100}) {
        std::vector<int> hits(n, 0);
        std::vector<int> bandOf(n, -1);
        // Results are recorded per band and checked here: CHECK counts
        // failures in a plain static, so it must not run on worker threads.
        std::vector<int> emptyBand(static_cast<size_t>(pool.bands()), 0);
        pool.parallelFor(0, n, [&](int band, int b, int e) {
            if (b >= e) emptyBand[static_cast<size_t>(band)] = 1;
            for (int i = b; i < e; ++i) { ++hits[i]; bandOf[i] = band; }
        });
        for (int v : emptyBand) CHECK_EQ(v, 0);
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
    std::atomic<int> sum(0);  // the pool is still usable afterwards
    pool.parallelFor(0, 8, [&](int, int b, int e) { sum += e - b; });
    CHECK_EQ(sum.load(), 8);
}

TEST_CASE("ThreadPool::parallelFor: an exception in the caller's own band (band 0) is rethrown too") {
    ThreadPool pool(2);
    bool caught = false;
    try {
        pool.parallelFor(0, 9, [](int band, int, int) { if (band == 0) throw std::runtime_error("band 0"); });
    } catch (const std::runtime_error&) { caught = true; }
    CHECK(caught);
    std::atomic<int> sum(0);
    pool.parallelFor(0, 9, [&](int, int b, int e) { sum += e - b; });
    CHECK_EQ(sum.load(), 9);
}

#endif // threads available

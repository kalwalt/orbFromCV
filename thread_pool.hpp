#ifndef ORB_THREAD_POOL_HPP
#define ORB_THREAD_POOL_HPP

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Fixed pool of std::threads that splits a row range into contiguous bands
// (issue #19). ORB::setNumThreads(n) owns one with n - 1 workers; the kernels
// take it as `ThreadPool* pool` and run their row loops through parallelFor,
// each band writing only its own rows, so the output is the same bytes in the
// same order as the sequential loop.
class ThreadPool {
public:
    // Starts `workers` (>= 1) threads; they are joined by the destructor. If
    // the system refuses a thread (std::system_error), the ones already
    // started are joined and the exception propagates.
    explicit ThreadPool(int workers);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int workers() const { return static_cast<int>(threads_.size()); }
    // Bands a parallelFor splits its range into (workers() + 1); kernels that
    // keep per-band state size it with this.
    int bands() const { return workers() + 1; }

    // Splits [begin, end) into workers() + 1 contiguous bands of nearly equal
    // size, band i being [begin + n*i/bands, begin + n*(i+1)/bands); the
    // caller runs band 0 itself. Empty bands are skipped. Calls
    // body(bandIndex, bandBegin, bandEnd) and returns when every band is
    // done. An exception thrown by a band is rethrown here (the first one
    // wins); the pool stays usable. One parallelFor at a time per pool: calls
    // are serialised by a mutex, so body must not call parallelFor on the
    // same pool (it would wait on itself).
    void parallelFor(int begin, int end, const std::function<void(int, int, int)>& body);

private:
    struct Job {
        const std::function<void(int, int, int)>* body;
        int begin, end, bands;
    };

    void workerLoop();
    void runBand(const Job& job, int band);

    std::vector<std::thread> threads_;
    std::mutex callMutex_;              // serialises parallelFor calls

    std::mutex mutex_;                  // guards everything below
    std::condition_variable wake_;      // workers wait for a job or for stop
    std::condition_variable done_;      // parallelFor waits for the bands
    Job job_{nullptr, 0, 0, 0};
    int nextBand_ = 0;                  // next band a worker may claim
    int finished_ = 0;                  // bands finished in this generation
    std::exception_ptr error_;
    bool stop_ = false;
};

#endif // ORB_THREAD_POOL_HPP

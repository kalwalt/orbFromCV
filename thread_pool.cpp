#include "thread_pool.hpp"
#include <algorithm>

ThreadPool::ThreadPool(int workers) {
    const int n = std::max(1, workers);
    threads_.reserve(static_cast<size_t>(n));
    try {
        for (int i = 0; i < n; ++i) threads_.emplace_back([this] { workerLoop(); });
    } catch (...) {
        // A joinable std::thread destroyed by the unwinding would terminate
        // the process; stop and join the workers that did start instead.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (std::thread& t : threads_) t.join();
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : threads_) t.join();
}

void ThreadPool::runBand(const Job& job, int band) {
    const long long n = static_cast<long long>(job.end) - job.begin;
    const int b = job.begin + static_cast<int>(n * band / job.bands);
    const int e = job.begin + static_cast<int>(n * (band + 1) / job.bands);
    if (b < e) {
        try {
            (*job.body)(band, b, e);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (++finished_ == job.bands) done_.notify_all();
}

void ThreadPool::workerLoop() {
    for (;;) {
        Job job;
        int band;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || nextBand_ < job_.bands; });
            if (stop_) return;
            job = job_;
            band = nextBand_++;
        }
        runBand(job, band);
    }
}

void ThreadPool::parallelFor(int begin, int end, const std::function<void(int, int, int)>& body) {
    std::lock_guard<std::mutex> call(callMutex_);
    if (end <= begin) return;

    const Job job{&body, begin, end, bands()};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
        nextBand_ = 1;      // band 0 is the caller's
        finished_ = 0;
        error_ = nullptr;
    }
    wake_.notify_all();

    runBand(job, 0);
    // Help with any band no worker has claimed yet, then wait for the rest.
    for (;;) {
        int band;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (nextBand_ >= job.bands) break;
            band = nextBand_++;
        }
        runBand(job, band);
    }

    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this, &job] { return finished_ == job.bands; });
        error = error_;
        error_ = nullptr;
        job_ = Job{nullptr, 0, 0, 0};
    }
    if (error) std::rethrow_exception(error);
}

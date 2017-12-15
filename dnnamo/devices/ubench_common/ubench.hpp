#ifndef _UBENCH_HPP_
#define _UBENCH_HPP_

#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

namespace ubench {

typedef std::chrono::microseconds duration_t;

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_, stop_time_;
    duration_t duration_;

    bool started_;
public:
    Timer() : started_(false) {
    }

    void start() {
        if (!started_) {
            start_time_ = std::chrono::high_resolution_clock::now();
            started_ = true;
        }
    }

    void stop() {
        if (started_) {
            stop_time_ = std::chrono::high_resolution_clock::now();
            duration_ += std::chrono::duration_cast<std::chrono::microseconds>(stop_time_ - start_time_);
            //duration_ += std::chrono::high_resolution_clock::now() - start_time_;
            //std::cout << "start time: " << start_time_ << "\n";
            //std::cout << "stop time: " << stop_time_ << "\n";
            //std::cout << duration_ << "----------";
            started_ = false;
        }
    }

    void reset() {
        started_ = false;
        duration_ = duration_t(0);
    }

    const duration_t& get_duration() const {
        return duration_;
    }
};

// Running state for a benchmark.
class State {
private:
    std::vector<duration_t> durations_;
public:
    void record(duration_t duration) {
        durations_.push_back(duration);
    }

    const std::vector<duration_t>& get_durations() const {
        return durations_;
    }

    duration_t get_mean_duration() const {
        double sum;

        for (auto duration : durations_) {
            sum += duration.count();
        }

        long long mean = static_cast<long long>(sum / durations_.size());

        return duration_t(mean);
    }
};

// TODO: return results rather than state
State run_benchmark(std::function<void()> f, size_t trials, size_t iterations) {
    State s;
    Timer t;

    t.reset();
    for (int trial = 0; trial < trials; trial++) {
        t.start();
        for (int i = 0; i < iterations; i++) {
            f();
        }
        t.stop();

        duration_t per_iteration = duration_t(t.get_duration().count() / iterations);
        s.record(per_iteration);

        t.reset();
    }

    return s;
}


}

#endif

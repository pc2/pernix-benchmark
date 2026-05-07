#include <CLI/CLI.hpp>

#include <membench/benchmark.h>

#include <iostream>

struct MemBenchConfig {
    
};

int main() {
    constexpr membench::BenchmarkConfig config = {
        8 * 1024,
        10000000,
        5,
        0,
        3
    };
    membench::BenchmarkContext ctx(config);

    membench::global_registry().run_all(ctx);

    return 0;
}

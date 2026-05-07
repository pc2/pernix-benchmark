#ifndef MEMBENCH_CPU_H
#define MEMBENCH_CPU_H

#include <string>

namespace membench {

struct PinResult {
    bool success;
    std::string message;
};

PinResult set_cpu_affinity(int cpu_id);

int get_cpu_count();

int get_cpu_id();

int reset_cpu_affinity();

} // namespace membench

#endif  // MEMBENCH_CPU_H

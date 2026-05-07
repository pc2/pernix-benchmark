#include <membench/cpu.h>

#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace membench {

PinResult set_cpu_affinity(const int cpu_id) {
    if (cpu_id < 0) {
        return {false, "cpu id must be non-negative"};
    }
    if (cpu_id >= get_cpu_count()) {
        return {false, "cpu id exceeds the number of available cpus"};
    }

#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    if (rc != 0) {
        return {
            false,
            std::string("pthread_setaffinity_np failed: ") + std::strerror(rc),
        };
    }

    return {
        true,
        "pinned current thread to cpu " + std::to_string(cpu_id),
    };
#elif defined(__APPLE__)
    // macOS does not provide strict Linux-Style CPU pinning.
    return {
            false,
            "macOS does not support setting CPU affinity",
    };
#elif defined(_WIN32)
    if (cpu_id >= static_cast<int>(sizeof(DWORD_PTR) * 8)) {
        return {
                false,
                "cpu id exceeds the number of available cpus",
        };
    }

    const DWORD_PTR mask = DWORD_PTR{1} << cpu_id;
    const DWORD_PTR old_mask = SetThreadAffinityMask(GetCurrentThread(), mask);

    if (old_mask == 0) {
        return {
                false,
                "SetThreadAffinityMask failed: " + std::to_string(GetLastError()),
        };
    }

    return {
        true,
        "set thread affinity to cpu " + std::to_string(cpu_id),
    };
#else
    (void)cpu_id;

    return {
        false,
        "unsupported platform for setting CPU affinity",
    };
#endif
}

int get_cpu_count() {
#if defined(__linux__)
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#elif defined(__APPLE__)
    int count;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.ncpu", &count, &size, nullptr, 0) != 0) {
        return -1; // Failed to get CPU count
    }
    return count;
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return static_cast<int>(sysinfo.dwNumberOfProcessors);
#else
    return -1; // Unsupported platform
#endif
}

int get_cpu_id() {
#if defined(__linux__)
    return sched_getcpu();
#elif defined(__APPLE__)
    // macOS does not provide a direct way to get the current CPU ID.
    // We can return -1 to indicate that this information is not available.
    return -1;
#elif defined(_WIN32)
    // Windows does not provide a direct way to get the current CPU ID.
    // We can return -1 to indicate that this information is not available.
    return -1;
#else
    return -1; // Unsupported platform
#endif
}

int reset_cpu_affinity() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < get_cpu_count(); ++i) {
        CPU_SET(i, &set);
    }

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    if (rc != 0) {
        return -1; // Failed to reset thread affinity
    }

    return 0;
#elif defined(__APPLE__)
    // macOS does not provide strict Linux-Style CPU pinning, so we can simply return success.
    return 0;
#elif defined(_WIN32)
    const DWORD_PTR mask = (1ULL << get_cpu_count()) - 1;
    const DWORD_PTR old_mask = SetThreadAffinityMask(GetCurrentThread(), mask);

    if (old_mask == 0) {
        return -1; // Failed to reset thread affinity
    }

    return 0;
#else
    return -1; // Unsupported platform
#endif
}

}  // namespace membench

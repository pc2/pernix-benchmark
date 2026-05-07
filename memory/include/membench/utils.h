#ifndef MEMBENCH_UTILS_H
#define MEMBENCH_UTILS_H
#include <type_traits>

#ifndef __always_inline
#if defined(__GNUC__) || defined(__clang__)
#define __always_inline inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define __always_inline __forceinline
#else
#define __always_inline inline
#endif
#endif

#if !defined(__GNUC__) || defined(__llvm__) || defined(__INTEL_COMPILER)
template <class Tp>
__always_inline void do_not_optimize(Tp const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
__always_inline void do_not_optimize(Tp& value) {
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

template <class Tp>
__always_inline void do_not_optimize(Tp&& value) {
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

#elif (__GNUC__ >= 5)
template <class Tp>
__always_inline std::enable_if_t<std::is_trivially_copyable_v<Tp> && (sizeof(Tp) <= sizeof(Tp*))>
    do_not_optimize(Tp const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
__always_inline std::enable_if_t<!std::is_trivially_copyable_v<Tp> || (sizeof(Tp) > sizeof(Tp*))>
    do_not_optimize(Tp const& value) {
    asm volatile("" : : "m"(value) : "memory");
}

template <class Tp>
__always_inline std::enable_if_t<std::is_trivially_copyable_v<Tp> && (sizeof(Tp) <= sizeof(Tp*))>
    do_not_optimize(Tp& value) {
    asm volatile("" : "+m,r"(value) : : "memory");
}

template <class Tp>
__always_inline std::enable_if_t<!std::is_trivially_copyable_v<Tp> || (sizeof(Tp) > sizeof(Tp*))>
    do_not_optimize(Tp& value) {
    asm volatile("" : "+m"(value) : : "memory");
}

template <class Tp>
__always_inline std::enable_if_t<std::is_trivially_copyable_v<Tp> && (sizeof(Tp) <= sizeof(Tp*))>
    do_not_optimize(Tp&& value) {
    asm volatile("" : "+m,r"(value) : : "memory");
}

template <class Tp>
__always_inline std::enable_if_t<!std::is_trivially_copyable_v<Tp> || (sizeof(Tp) > sizeof(Tp*))>
    do_not_optimize(Tp&& value) {
    asm volatile("" : "+m"(value) : : "memory");
}
#endif

#endif  // MEMBENCH_UTILS_H

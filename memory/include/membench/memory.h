#ifndef MEMBENCH_MEMORY_H
#define MEMBENCH_MEMORY_H

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>

namespace membench {

constexpr size_t default_alignment = 64;

template<class T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(const size_t size, size_t alignment = default_alignment): size_(size) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("alignment must be a non-zero power of two");
        }
        if (alignment < sizeof(void*)) {
            alignment = sizeof(void*);
        }
        if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length{};
        }

        const size_t bytes = size * sizeof(T);
        if (bytes == 0) {
            return;
        }

#if defined(_MSC_VER)
        ptr_ = static_cast<T*>(_aligned_malloc(bytes, alignment));
        if (!ptr_) {
            throw std::bad_alloc();
        }
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            throw std::bad_array_new_length{};
        }
        const size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);
        ptr_ = static_cast<T*>(aligned_alloc(alignment, aligned_bytes));
        if (!ptr_) {
            throw std::bad_alloc();
        }
#else
        void* raw = nullptr;
        if (posix_memalign(&raw, alignment, bytes) != 0) {
            throw std::bad_alloc{};
        }
        ptr_ = static_cast<T*>(raw);
#endif
    }

    ~AlignedBuffer() {
        if (ptr_) {
#if defined(_MSC_VER)
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
        }
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        if (ptr_) {
#if defined(_MSC_VER)
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
        }

        ptr_ = other.ptr_;
        size_ = other.size_;

        other.ptr_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    [[nodiscard]] T* data() noexcept {
        return ptr_;
    }

    [[nodiscard]] const T* data() const noexcept {
        return ptr_;
    }

    [[nodiscard]] size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] std::span<T> span() noexcept {
        return std::span<T>(ptr_, size_);
    }

    [[nodiscard]] std::span<const T> span() const noexcept {
        return std::span<const T>(ptr_, size_);
    }

    void fill(T value) {
        if (size_ == 0) {
            return;
        }
        std::fill(ptr_, ptr_ + size_, value);
    }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

} // namespace membench

#endif  // MEMBENCH_MEMORY_H

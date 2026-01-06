module cp2k_benchmarks
    use kinds
    use hfx_compression_methods
    use hfx_types
    use omp_lib
    use iso_c_binding
    implicit none
contains

    function cp2k_compression(user_state, n_iters, n_bits, n_values) bind(C, name = "cp2k_compression") result(elapsed_time)
        type(c_ptr), value :: user_state
        integer(c_int), value :: n_iters, n_values, n_bits

        integer(c_int) :: i
        integer(c_int) :: memory_usage_bytes
        type(hfx_compression_type) :: compression_store
        type(hfx_container_type), dimension(:), pointer :: integral_containers
        type(hfx_cache_type), dimension(:), pointer :: integral_caches
        type(hfx_container_type), pointer :: maxval_container
        type(hfx_cache_type), pointer :: maxval_cache

        logical :: use_disk_storage = .false.
        real(dp) :: eps_storage
        real(dp) :: b_max
        real(dp), parameter :: c_max = 1.0_dp  ! maximal absolute contraction coefficient
        real(dp), parameter :: p_max_entry = 1.0_dp
        real(c_double) :: t_start, t_end, elapsed_time

        real(dp), allocatable, dimension(:) :: values_in, values_ref

        ! Mark unused ABI parameters to silence warnings; keep signature stable.
        if (c_associated(user_state)) then
            ! no-op
        end if

        ! Defaults
        elapsed_time = -1.0_c_double

        ! Basic validation (C-friendly: return negative on error)
        if (n_iters <= 0_c_int) return
        if (n_values <= 0_c_int) return

        allocate(values_in(n_values))
        allocate(values_ref(n_values))

        call alloc_containers(compression_store, 1)

        maxval_container => compression_store%maxval_container(1)
        maxval_cache => compression_store%maxval_cache(1)
        integral_containers => compression_store%integral_containers(:, 1)
        integral_caches => compression_store%integral_caches(:, 1)

        call hfx_init_container(maxval_container, memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(maxval_cache, maxval_container, memory_usage_bytes, .false.)

        do i = 1, n_values
            values_ref(i) = sin(real(i, dp))
        end do

        if (n_bits < 1_c_int)  n_bits = 1_c_int
        if (n_bits > 64_c_int) n_bits = 64_c_int

        ! compute b_max as before
        b_max = maxval(abs(values_ref))

        ! compute eps_storage from eps = b_max / (2^(n_bits-1) - 1)
        if (b_max <= 0.0_dp) then
            eps_storage = 0.0_dp
        else if (n_bits <= 1_c_int) then
            ! n=1 -> denominator is zero; treat as no fractional precision
            eps_storage = b_max
        else
            eps_storage = b_max / (2.0_dp**(real(n_bits - 1_c_int, kind = dp)) - 1.0_dp)
        end if

        ! Prevent out-of-bounds accesses to (1:64)
        if (n_bits < 1_c_int)  n_bits = 1_c_int
        if (n_bits > 64_c_int) n_bits = 64_c_int

        do i = 1, n_bits
            call hfx_init_container(integral_containers(i), memory_usage_bytes, .false.)
            call hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), memory_usage_bytes, .false.)
        end do

        ! Warm-up (not timed)
        values_in(:) = values_ref(:)
        call hfx_add_mult_cache_elements(&
                values_in, n_values, n_bits, &
                integral_caches(n_bits), integral_containers(n_bits), &
                eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)

        t_start = real(omp_get_wtime(), kind = c_double)
        do i = 1, n_iters
            values_in(:) = values_ref(:)
            call hfx_add_mult_cache_elements(&
                    values_in, n_values, n_bits, &
                    integral_caches(n_bits), integral_containers(n_bits), &
                    eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
        end do
        t_end = real(omp_get_wtime(), kind = c_double)

        do i = 1, n_bits
            call hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), memory_usage_bytes, .false.)
        end do

        call dealloc_containers(compression_store, memory_usage_bytes)

        if (allocated(values_in))  deallocate(values_in)
        if (allocated(values_ref)) deallocate(values_ref)

        elapsed_time = t_end - t_start
    end function cp2k_compression

    function cp2k_decompression(user_state, n_iters, n_bits, n_values) bind(C, name = "cp2k_decompression") result(elapsed_time)
        type(c_ptr), value :: user_state
        integer(c_int), value :: n_iters, n_values, n_bits

        integer(c_int) :: i
        integer(c_int) :: memory_usage_bytes
        type(hfx_compression_type) :: compression_store
        type(hfx_container_type), dimension(:), pointer :: integral_containers
        type(hfx_cache_type), dimension(:), pointer :: integral_caches
        type(hfx_container_type), pointer :: maxval_container
        type(hfx_cache_type), pointer :: maxval_cache

        logical :: use_disk_storage = .false.
        real(dp) :: eps_storage
        real(dp) :: b_max
        real(dp), parameter :: c_max = 1.0_dp  ! maximal absolute contraction coefficient
        real(dp), parameter :: p_max_entry = 1.0_dp
        real(c_double) :: t_start, t_end, elapsed_time

        real(dp), allocatable, dimension(:) :: values_in, values_ref, values_out

        ! Mark unused ABI parameters to silence warnings; keep signature stable.
        if (c_associated(user_state)) then
            ! no-op
        end if

        ! Defaults
        elapsed_time = -1.0_c_double

        ! Basic validation (C-friendly: return negative on error)
        if (n_iters <= 0_c_int) return
        if (n_values <= 0_c_int) return

        allocate(values_in(n_values))
        allocate(values_ref(n_values))
        allocate(values_out(n_values))

        call alloc_containers(compression_store, 1)

        maxval_container => compression_store%maxval_container(1)
        maxval_cache => compression_store%maxval_cache(1)
        integral_containers => compression_store%integral_containers(:, 1)
        integral_caches => compression_store%integral_caches(:, 1)

        call hfx_init_container(maxval_container, memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(maxval_cache, maxval_container, memory_usage_bytes, .false.)

        do i = 1, n_values
            values_ref(i) = sin(real(i, dp))
        end do

        if (n_bits < 1_c_int)  n_bits = 1_c_int
        if (n_bits > 64_c_int) n_bits = 64_c_int

        ! compute b_max as before
        b_max = maxval(abs(values_ref))

        ! compute eps_storage from eps = b_max / (2^(n_bits-1) - 1)
        if (b_max <= 0.0_dp) then
            eps_storage = 0.0_dp
        else if (n_bits <= 1_c_int) then
            ! n=1 -> denominator is zero; treat as no fractional precision
            eps_storage = b_max
        else
            eps_storage = b_max / (2.0_dp**(real(n_bits - 1_c_int, kind = dp)) - 1.0_dp)
        end if

        ! Prevent out-of-bounds accesses to (1:64)
        if (n_bits < 1_c_int)  n_bits = 1_c_int
        if (n_bits > 64_c_int) n_bits = 64_c_int

        do i = 1, n_bits
            call hfx_init_container(integral_containers(i), memory_usage_bytes, .false.)
            call hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), memory_usage_bytes, .false.)
        end do

        do i = 1, n_iters
            values_in(:) = values_ref(:)
            call hfx_add_mult_cache_elements(&
                    values_in, n_values, n_bits, &
                    integral_caches(n_bits), integral_containers(n_bits), &
                    eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
        end do

        !flush is not required
        DO i = 1, n_bits
            CALL hfx_flush_last_cache(i, integral_caches(i), integral_containers(i), memory_usage_bytes, .FALSE.)
        END DO
        !        print*,"memory_usage_bytes",memory_usage_bytes

        DO i = 1, n_bits
            CALL hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), memory_usage_bytes, .FALSE.)
        END DO

        DO i = 1, n_bits
            CALL hfx_decompress_first_cache(i, integral_caches(i), integral_containers(i), memory_usage_bytes, .FALSE.)
        END DO

        ! Warm-up (not timed)
        call hfx_get_mult_cache_elements(&
                values_out, n_values, n_bits, &
                integral_caches(n_bits), integral_containers(n_bits), &
                eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)

        t_start = real(omp_get_wtime(), kind = c_double)
        do i = 1, n_iters
            CALL hfx_get_mult_cache_elements(&
                    values_out, n_values, n_bits, &
                    integral_caches(n_bits), integral_containers(n_bits), &
                    eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
        end do
        t_end = real(omp_get_wtime(), kind = c_double)

        do i = 1, n_bits
            call hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), memory_usage_bytes, .false.)
        end do

        call dealloc_containers(compression_store, memory_usage_bytes)

        if (allocated(values_in))  deallocate(values_in)
        if (allocated(values_ref)) deallocate(values_ref)
        if (allocated(values_out)) deallocate(values_out)

        elapsed_time = t_end - t_start
    end function cp2k_decompression

end module cp2k_benchmarks


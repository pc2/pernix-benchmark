module cp2k_benchmarks
    use kinds
    use hfx_compression_methods
    use hfx_types
    use omp_lib
    use iso_c_binding
    implicit none
contains

    function cp2k_compression(user_state, n_iters, n_bits, blocks) bind(C, name = "cp2k_compression") result(elapsed_time)
        type(c_ptr), value :: user_state
        integer(c_int), value :: n_iters, blocks, n_bits
        integer(c_int) :: elements_per_blocks, elements_total
        integer(c_int) :: i, j
        integer(c_int) :: memory_usage_bytes
        type(hfx_compression_type) :: compression_store
        type(hfx_container_type), dimension(:), pointer :: integral_containers
        type(hfx_cache_type), dimension(:), pointer :: integral_caches
        type(hfx_container_type), pointer :: maxval_container
        type(hfx_cache_type), pointer :: maxval_cache
        logical :: use_disk_storage = .false.
        real(dp) :: eps_storage
        real(dp) :: b_max
        real(dp), parameter :: p_max_entry = 1.0_dp
        real(c_double) :: t_start, t_end, elapsed_time
        real(dp), allocatable, dimension(:) :: values_in
        logical :: containers_allocated, values_allocated

        containers_allocated = .false.
        values_allocated = .false.

        if (c_associated(user_state)) then
            ! no-op
        end if

        elapsed_time = -1.0_c_double
        memory_usage_bytes = 0_c_int

        if (n_bits < 1_c_int)  n_bits = 1_c_int
        if (n_bits > 64_c_int) n_bits = 64_c_int
        if (n_iters <= 0_c_int) goto 999
        if (blocks <= 0_c_int) goto 999

        elements_per_blocks = 512 / n_bits
        elements_total = elements_per_blocks * blocks
        if (elements_total <= 0_c_int) goto 999

        allocate(values_in(elements_total))
        values_allocated = .true.

        call alloc_containers(compression_store, 1)
        containers_allocated = .true.

        maxval_container => compression_store%maxval_container(1)
        maxval_cache => compression_store%maxval_cache(1)
        integral_containers => compression_store%integral_containers(:, 1)
        integral_caches => compression_store%integral_caches(:, 1)

        call hfx_init_container(maxval_container, memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(maxval_cache, maxval_container, memory_usage_bytes, .false.)

        do i = 1, elements_total
            values_in(i) = sin(real(i, dp))
        end do

        b_max = maxval(abs(values_in))

        if (b_max <= 0.0_dp) then
            eps_storage = 0.0_dp
        else if (n_bits <= 1_c_int) then
            eps_storage = b_max
        else
            eps_storage = b_max / (2.0_dp**(real(n_bits - 1_c_int, kind = dp)) - 1.0_dp)
        end if
        if (eps_storage <= 0.0_dp) eps_storage = epsilon(1.0_dp)

        call hfx_init_container(integral_containers(n_bits), memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)

        do j = 1, blocks
            call hfx_add_mult_cache_elements(values_in, elements_per_blocks, n_bits, &
                    integral_caches(n_bits), integral_containers(n_bits), &
                    eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
        end do

        t_start = real(omp_get_wtime(), kind = c_double)
        do i = 1, n_iters
            !            integral_caches(n_bits)%element_counter = 1
            !            integral_containers(n_bits)%element_counter = 1
            !            integral_containers(n_bits)%file_counter = 1
            !            integral_containers(n_bits)%current => integral_containers(n_bits)%first
            do j = 1, blocks
                call hfx_add_mult_cache_elements(values_in, elements_per_blocks, n_bits, &
                        integral_caches(n_bits), integral_containers(n_bits), &
                        eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
            end do
        end do
        t_end = real(omp_get_wtime(), kind = c_double)

        elapsed_time = t_end - t_start

        999     continue
        if (containers_allocated) then
            call hfx_flush_last_cache(n_bits, integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)
            call hfx_reset_cache_and_container(integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)
            call hfx_reset_cache_and_container(maxval_cache, maxval_container, memory_usage_bytes, .false.)
            call dealloc_containers(compression_store, memory_usage_bytes)
        end if
        if (values_allocated) then
            deallocate(values_in)
        end if
    end function cp2k_compression

    function cp2k_decompression(user_state, n_iters, n_bits, blocks) bind(C, name = "cp2k_decompression") result(elapsed_time)
        type(c_ptr), value :: user_state
        integer(c_int), value :: n_iters, blocks, n_bits
        integer(c_int) :: elements_per_blocks, elements_total
        integer(c_int) :: i, j, memory_usage_bytes
        type(hfx_compression_type) :: compression_store
        type(hfx_container_type), dimension(:), pointer :: integral_containers
        type(hfx_cache_type), dimension(:), pointer :: integral_caches
        type(hfx_container_type), pointer :: maxval_container
        type(hfx_cache_type), pointer :: maxval_cache
        logical :: use_disk_storage = .false.
        real(dp) :: eps_storage
        real(dp) :: b_max
        real(dp), parameter :: p_max_entry = 1.0_dp
        real(c_double) :: t_start, t_end, elapsed_time
        real(dp), allocatable, dimension(:) :: values_in, values_out
        logical :: containers_allocated, values_allocated

        containers_allocated = .false.
        values_allocated = .false.

        if (c_associated(user_state)) then
            ! no-op
        end if

        elapsed_time = -1.0_c_double
        memory_usage_bytes = 0_c_int

        if (n_bits < 1_c_int)  n_bits = 1_c_int
        if (n_bits > 64_c_int) n_bits = 64_c_int
        if (n_iters <= 0_c_int) goto 999
        if (blocks <= 0_c_int) goto 999

        elements_per_blocks = 512 / n_bits
        elements_total = elements_per_blocks * blocks
        if (elements_total <= 0_c_int) goto 999

        allocate(values_in(elements_total))
        allocate(values_out(elements_total))
        values_allocated = .true.

        call alloc_containers(compression_store, 1)
        containers_allocated = .true.

        maxval_container => compression_store%maxval_container(1)
        maxval_cache => compression_store%maxval_cache(1)
        integral_containers => compression_store%integral_containers(:, 1)
        integral_caches => compression_store%integral_caches(:, 1)

        call hfx_init_container(maxval_container, memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(maxval_cache, maxval_container, memory_usage_bytes, .false.)

        do i = 1, elements_total
            values_in(i) = sin(real(i, dp))
        end do

        b_max = maxval(abs(values_in))

        if (b_max <= 0.0_dp) then
            eps_storage = 0.0_dp
        else if (n_bits <= 1_c_int) then
            eps_storage = b_max
        else
            eps_storage = b_max / (2.0_dp**(real(n_bits - 1_c_int, kind = dp)) - 1.0_dp)
        end if
        if (eps_storage <= 0.0_dp) eps_storage = epsilon(1.0_dp)

        call hfx_init_container(integral_containers(n_bits), memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)

        do j = 1, blocks
            call hfx_add_mult_cache_elements(values_in, elements_per_blocks, n_bits, &
                    integral_caches(n_bits), integral_containers(n_bits), &
                    eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
        end do

        call hfx_flush_last_cache(n_bits, integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)
        call hfx_reset_cache_and_container(integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)
        call hfx_decompress_first_cache(n_bits, integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)

        do j = 1, blocks
            call hfx_get_mult_cache_elements(values_out, elements_per_blocks, n_bits, &
                    integral_caches(n_bits), integral_containers(n_bits), &
                    eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
        end do

        t_start = real(omp_get_wtime(), kind = c_double)
        do i = 1, n_iters
            integral_caches(n_bits)%element_counter = 1
            integral_containers(n_bits)%element_counter = 1
            integral_containers(n_bits)%file_counter = 1
            integral_containers(n_bits)%current => integral_containers(n_bits)%first
            do j = 1, blocks
                call hfx_get_mult_cache_elements(values_out, elements_per_blocks, n_bits, &
                        integral_caches(n_bits), integral_containers(n_bits), &
                        eps_storage, p_max_entry, memory_usage_bytes, use_disk_storage)
            end do
        end do
        t_end = real(omp_get_wtime(), kind = c_double)

        elapsed_time = t_end - t_start

        999     continue
        if (containers_allocated) then
            call hfx_reset_cache_and_container(integral_caches(n_bits), integral_containers(n_bits), memory_usage_bytes, .false.)
            call hfx_reset_cache_and_container(maxval_cache, maxval_container, memory_usage_bytes, .false.)
            call dealloc_containers(compression_store, memory_usage_bytes)
        end if
        if (values_allocated) then
            deallocate(values_in)
            deallocate(values_out)
        end if
    end function cp2k_decompression

end module cp2k_benchmarks
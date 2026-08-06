module cp2k_benchmark_adapter
    use hfx_compression_core_methods, only: bits2ints_specific, ints2bits_specific
    use iso_c_binding, only: c_double, c_float, c_int, c_int64_t, c_long_long
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none

    integer, parameter :: block_bits = 512
    integer, parameter :: max_elements_per_block = block_bits

    interface
        function c_llrint(value) bind(C, name="llrint") result(rounded)
            import c_double, c_long_long
            real(c_double), value :: value
            integer(c_long_long) :: rounded
        end function c_llrint

        function c_llrintf(value) bind(C, name="llrintf") result(rounded)
            import c_float, c_long_long
            real(c_float), value :: value
            integer(c_long_long) :: rounded
        end function c_llrintf
    end interface

contains

    subroutine cp2k_compress_block_f32(input, scale, packed, bit_width) bind(C, name="cp2k_compress_block_f32")
        real(c_float), intent(in) :: input(*)
        real(c_float), value :: scale
        integer(c_int64_t), intent(out) :: packed(*)
        integer(c_int), value :: bit_width

        integer(c_int64_t) :: full_data(max_elements_per_block)
        integer(c_int64_t) :: encoded, quantized, range_size
        integer :: elements, i
        real(c_float) :: lower_bound, scaled, upper_bound

        if (bit_width < 1_c_int .or. bit_width > 24_c_int) return

        elements = block_bits / bit_width
        range_size = shiftl(1_c_int64_t, bit_width)
        if (bit_width == 1_c_int) then
            lower_bound = 0.0_c_float
            upper_bound = 1.0_c_float
        else
            lower_bound = -real(range_size / 2_c_int64_t, c_float)
            upper_bound = real(range_size / 2_c_int64_t - 1_c_int64_t, c_float)
        end if

        do i = 1, elements
            scaled = input(i) * scale
            if (ieee_is_nan(scaled)) then
                quantized = 0_c_int64_t
            else
                quantized = c_llrintf(max(lower_bound, min(upper_bound, scaled)))
            end if

            encoded = quantized
            if (encoded < 0_c_int64_t) encoded = encoded + range_size
            full_data(i) = encoded
        end do

        call ints2bits_specific(bit_width, elements, packed, full_data)
    end subroutine cp2k_compress_block_f32

    subroutine cp2k_decompress_block_f32(packed, scale, output, bit_width) bind(C, name="cp2k_decompress_block_f32")
        integer(c_int64_t), intent(in) :: packed(*)
        real(c_float), value :: scale
        real(c_float), intent(out) :: output(*)
        integer(c_int), value :: bit_width

        integer(c_int64_t) :: full_data(max_elements_per_block)
        integer(c_int64_t) :: range_size, sign_bit, value
        integer :: elements, i

        if (bit_width < 1_c_int .or. bit_width > 24_c_int) return

        elements = block_bits / bit_width
        call bits2ints_specific(bit_width, elements, packed, full_data)

        range_size = shiftl(1_c_int64_t, bit_width)
        sign_bit = range_size / 2_c_int64_t
        do i = 1, elements
            value = full_data(i)
            if (bit_width > 1_c_int .and. value >= sign_bit) value = value - range_size
            output(i) = real(value, c_float) * scale
        end do
    end subroutine cp2k_decompress_block_f32

    subroutine cp2k_compress_block_f64(input, scale, packed, bit_width) bind(C, name="cp2k_compress_block_f64")
        real(c_double), intent(in) :: input(*)
        real(c_double), value :: scale
        integer(c_int64_t), intent(out) :: packed(*)
        integer(c_int), value :: bit_width

        integer(c_int64_t) :: full_data(max_elements_per_block)
        integer(c_int64_t) :: encoded, quantized, range_size
        integer :: elements, i
        real(c_double) :: lower_bound, scaled, upper_bound

        if (bit_width < 1_c_int .or. bit_width > 24_c_int) return

        elements = block_bits / bit_width
        range_size = shiftl(1_c_int64_t, bit_width)
        if (bit_width == 1_c_int) then
            lower_bound = 0.0_c_double
            upper_bound = 1.0_c_double
        else
            lower_bound = -real(range_size / 2_c_int64_t, c_double)
            upper_bound = real(range_size / 2_c_int64_t - 1_c_int64_t, c_double)
        end if

        do i = 1, elements
            scaled = input(i) * scale
            if (ieee_is_nan(scaled)) then
                quantized = 0_c_int64_t
            else
                quantized = c_llrint(max(lower_bound, min(upper_bound, scaled)))
            end if

            encoded = quantized
            if (encoded < 0_c_int64_t) encoded = encoded + range_size
            full_data(i) = encoded
        end do

        call ints2bits_specific(bit_width, elements, packed, full_data)
    end subroutine cp2k_compress_block_f64

    subroutine cp2k_decompress_block_f64(packed, scale, output, bit_width) bind(C, name="cp2k_decompress_block_f64")
        integer(c_int64_t), intent(in) :: packed(*)
        real(c_double), value :: scale
        real(c_double), intent(out) :: output(*)
        integer(c_int), value :: bit_width

        integer(c_int64_t) :: full_data(max_elements_per_block)
        integer(c_int64_t) :: range_size, sign_bit, value
        integer :: elements, i

        if (bit_width < 1_c_int .or. bit_width > 24_c_int) return

        elements = block_bits / bit_width
        call bits2ints_specific(bit_width, elements, packed, full_data)

        range_size = shiftl(1_c_int64_t, bit_width)
        sign_bit = range_size / 2_c_int64_t
        do i = 1, elements
            value = full_data(i)
            if (bit_width > 1_c_int .and. value >= sign_bit) value = value - range_size
            output(i) = real(value, c_double) * scale
        end do
    end subroutine cp2k_decompress_block_f64

end module cp2k_benchmark_adapter

! CP2K: A general program to perform molecular dynamics simulations.
! Copyright 2026 CP2K developers group <https://cp2k.org>
! SPDX-License-Identifier: GPL-2.0-or-later
!
! C bindings for the original fixed-width CP2K HFX integer codec.
MODULE hfx_original_codec
   USE hfx_compression_core_methods, ONLY: bits2ints_specific, ints2bits_specific
   USE kinds, ONLY: int_8
   USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_INT64_T, C_SIZE_T
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: cp2k_hfx_pack_fixed, cp2k_hfx_unpack_fixed
CONTAINS
   FUNCTION cp2k_hfx_pack_fixed(values, ndata, nbits, packed) RESULT(nwords) &
      BIND(C, NAME="cp2k_hfx_pack_fixed")
      INTEGER(C_INT64_T), INTENT(IN) :: values(*)
      INTEGER(C_INT64_T), INTENT(OUT) :: packed(*)
      INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: ndata
      INTEGER(C_INT), VALUE, INTENT(IN) :: nbits
      INTEGER(C_SIZE_T) :: nwords
      INTEGER(int_8) :: encoded(1024), offset
      INTEGER :: count, width

      nwords = 0_C_SIZE_T
      IF (ndata == 0_C_SIZE_T .OR. ndata > 1024_C_SIZE_T .OR. nbits < 1_C_INT .OR. nbits > 24_C_INT) RETURN
      count = INT(ndata)
      width = INT(nbits)
      offset = SHIFTL(1_int_8, width - 1)
      encoded(1:count) = INT(values(1:count), int_8) + offset
      CALL ints2bits_specific(width, count, packed, encoded)
      nwords = (ndata * INT(nbits, C_SIZE_T) + 63_C_SIZE_T) / 64_C_SIZE_T
   END FUNCTION cp2k_hfx_pack_fixed

   FUNCTION cp2k_hfx_unpack_fixed(packed, ndata, nbits, values) RESULT(nwords) &
      BIND(C, NAME="cp2k_hfx_unpack_fixed")
      INTEGER(C_INT64_T), INTENT(IN) :: packed(*)
      INTEGER(C_INT64_T), INTENT(OUT) :: values(*)
      INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: ndata
      INTEGER(C_INT), VALUE, INTENT(IN) :: nbits
      INTEGER(C_SIZE_T) :: nwords
      INTEGER(int_8) :: decoded(1024), offset
      INTEGER :: count, width

      nwords = 0_C_SIZE_T
      IF (ndata == 0_C_SIZE_T .OR. ndata > 1024_C_SIZE_T .OR. nbits < 1_C_INT .OR. nbits > 24_C_INT) RETURN
      count = INT(ndata)
      width = INT(nbits)
      offset = SHIFTL(1_int_8, width - 1)
      CALL bits2ints_specific(width, count, packed, decoded)
      values(1:count) = INT(decoded(1:count) - offset, C_INT64_T)
      nwords = (ndata * INT(nbits, C_SIZE_T) + 63_C_SIZE_T) / 64_C_SIZE_T
   END FUNCTION cp2k_hfx_unpack_fixed
END MODULE hfx_original_codec

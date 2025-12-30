!Program test
!    use kinds
!    use hfx_compression_methods
!    use hfx_types
!    use omp_lib
!    implicit none
!
!    Integer(int_8) :: value,value2
!    integer(4) :: nbits,my_bin_size,bin,actual_memory_usage,nv,i,niter,iter
!    logical(4) :: use_disk_storage=.false.
!    TYPE(hfx_compression_type) :: store_ints
!    TYPE(hfx_container_type), DIMENSION(:), POINTER    :: integral_containers
!    TYPE(hfx_cache_type), DIMENSION(:), POINTER        :: integral_caches
!    TYPE(hfx_container_type), POINTER                  :: maxval_container
!    TYPE(hfx_cache_type), POINTER                  :: maxval_cache
!    INTEGER(int_8) :: max_val_memory
!    real(dp) :: t0,t1,diff2
!    real(dp) :: TW,TR
!    real(dp) :: MBraw,MBcomp
!
!    REAL(dp), ALLOCATABLE, DIMENSION(:) :: valuesin,valuesout,valuesin2
!    real(dp) :: eps_storage=1e-5
!    real(dp) :: pmax_entry=1.0D0 !maximal absolute density matrix element
!    real(dp) :: bmax
!    real(dp) :: cmax=1.0D0 !maximal absolute contraction coefficient
!
!    integer(4) :: eps_bits
!
!    my_bin_size=1
!
!    max_val_memory = 1_int_8
!
!    nv=1024 !CACHE_SIZE or smaller
!    niter=1000000
!    allocate(valuesin(nv))
!    allocate(valuesin2(nv))
!    allocate(valuesout(nv))
!    do eps_bits=10,20
!        CALL alloc_containers(store_ints, my_bin_size)
!        DO bin = 1, my_bin_size
!            maxval_container => store_ints%maxval_container(bin)
!            maxval_cache => store_ints%maxval_cache(bin)
!            integral_containers => store_ints%integral_containers(:, bin)
!            integral_caches => store_ints%integral_caches(:, bin)
!
!            CALL hfx_init_container(maxval_container, actual_memory_usage, .FALSE.)
!            DO i = 1, 64
!                CALL hfx_init_container(integral_containers(i), actual_memory_usage, .FALSE.)
!            END DO
!            CALL hfx_reset_cache_and_container(maxval_cache, maxval_container, actual_memory_usage, .FALSE.)
!            DO i = 1, 64
!                CALL hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), actual_memory_usage, .FALSE.)
!            END DO
!        END DO
!
!        eps_storage=2.0D0**(-eps_bits)
!        do i=1,nv
!            valuesin2(i)=sin(real(i,kind=8))
!        enddo
!        valuesout(:)=0.0D0
!
!        bmax=maxval(abs(valuesin2))
!        nbits=EXPONENT(ANINT(bmax*cmax*pmax_entry/eps_storage)) + 1
!        print*,"##################################################################"
!        print*,"eps_storage",eps_storage
!        print*,"",nbits
!        print*,"bmax",bmax
!        MBraw=real(nv,kind=8)*real(niter,kind=8)*8/1000000.0
!        MBcomp=real(nv,kind=8)*real(niter,kind=8)*real(nbits,kind=8)/8/1000000.0
!
!        DO bin = 1, my_bin_size
!            integral_containers => store_ints%integral_containers(:, bin)
!            integral_caches => store_ints%integral_caches(:, bin)
!        enddo
!
!        t0=omp_get_wtime()
!        do iter=1,niter
!            valuesin(:)=valuesin2(:)
!            CALL hfx_add_mult_cache_elements(valuesin,nv,nbits,integral_caches(nbits),integral_containers(nbits),eps_storage,pmax_entry,actual_memory_usage,use_disk_storage)
!        enddo
!        t1=omp_get_wtime()
!        TW=t1-t0
!        print*,"time fo writing",TW,nv*niter,MBraw/TW,MBcomp/TW
!
!        !flush is not required
!        DO i = 1, 64
!            CALL hfx_flush_last_cache(i, integral_caches(i), integral_containers(i), actual_memory_usage, .FALSE.)
!        END DO
!        print*,"actual_memory_usage",actual_memory_usage
!
!        DO i = 1, 64
!            CALL hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), actual_memory_usage, .FALSE.)
!        END DO
!
!        DO i = 1, 64
!            CALL hfx_decompress_first_cache(i, integral_caches(i), integral_containers(i), actual_memory_usage, .FALSE.)
!        END DO
!
!        t0=omp_get_wtime()
!
!        diff2=0
!        do iter=1,niter
!            CALL hfx_get_mult_cache_elements(valuesout,nv,nbits,integral_caches(nbits),integral_containers(nbits),eps_storage,pmax_entry,actual_memory_usage,use_disk_storage)
!            !diff2=max(diff2,sum(abs(valuesin2-valuesout))/real(nv,kind=8))
!        enddo
!        !print*,"diff2",diff2
!
!        t1=omp_get_wtime()
!        TR=t1-t0
!        print*,"time fo reading",TR,nv*niter,MBRAW/TR,MBcomp/TR
!
!        !    do i=1,nv
!        !      call hfx_add_single_cache_element(valuesin(i), nbits, maxval_cache, maxval_container, actual_memory_usage, use_disk_storage, max_val_memory)
!        !    enddo
!        !
!        !    do i=1,nv
!        !      call hfx_get_single_cache_element(valuesout(i), nbits, maxval_cache, maxval_container, actual_memory_usage, use_disk_storage)
!        !    enddo
!
!        print*,"diff",sum(abs(valuesin2-valuesout))/real(nv,kind=8)
!        print*,"total memory for uncompressed in MB",MBraw
!        print*,"total memory for compressed in MB  ",MBcomp
!
!        ! reset containers, TODO: does not deallocate, only reset positions and counters?
!        DO i = 1, 64
!            CALL hfx_reset_cache_and_container(integral_caches(i), integral_containers(i), actual_memory_usage, .FALSE.)
!        END DO
!
!        CALL dealloc_containers(store_ints,actual_memory_usage)
!    enddo
!
!    if (allocated(valuesin))    deallocate(valuesin)
!    if (allocated(valuesin2))   deallocate(valuesin2)
!    if (allocated(valuesout))   deallocate(valuesout)
!End program

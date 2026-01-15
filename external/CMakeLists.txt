#add_subdirectory(/home/woobb/Projects/libcompression libcompression)
add_subdirectory(pernix)

# apply patch to CP2K source files to build a static library for compression methods

set(CP2K_SOURCE_FILES
        cp2k/src/base/machine_cpuid.c

        cp2k/src/base/kinds.F
        cp2k/src/hfx_types.F
        cp2k/src/base/machine.F
        cp2k/src/base/base_hooks.F
        cp2k/src/common/cp_files.F
        cp2k/src/hfxbase/hfx_compression_core_methods.F
        cp2k/src/hfx_compression_methods.F
)

add_library(cp2k_compression STATIC ${CP2K_SOURCE_FILES})
target_compile_options(cp2k_compression PRIVATE -fopenmp)
target_compile_options(cp2k_compression PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-ffree-form -ffree-line-length-none -g -march=native>)
target_compile_definitions(cp2k_compression PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:__SHORT_FILE__=\"__FILE__\">
        __DATA_DIR=""
)
set_target_properties(cp2k_compression PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran_modules/cp2k"
)
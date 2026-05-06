#add_subdirectory(/home/woobb/Projects/libcompression libcompression)
# add_subdirectory(pernix)

# CP2K is GPL-2.0-or-later. This target compiles selected CP2K source files into a
# static library, so cp2k_compression and executables linked against it are treated
# as GPL-2.0-or-later. See ../LICENSES.md.
# Apply a repository patch to CP2K source files before building the compression methods.

find_package(Git REQUIRED)
find_package(OpenMP)

set(CP2K_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cp2k")
set(CP2K_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cp2k-patches/0001-Delete-unneeded-types-hfx_types.F.patch")

if (NOT EXISTS "${CP2K_SOURCE_DIR}/src/hfx_types.F")
        message(FATAL_ERROR "CP2K source directory not found or incomplete: ${CP2K_SOURCE_DIR}")
endif ()

if (NOT EXISTS "${CP2K_PATCH_FILE}")
        message(FATAL_ERROR "CP2K patch file not found: ${CP2K_PATCH_FILE}")
endif ()

execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check "${CP2K_PATCH_FILE}"
        WORKING_DIRECTORY "${CP2K_SOURCE_DIR}"
        RESULT_VARIABLE CP2K_PATCH_CAN_APPLY
        OUTPUT_QUIET
        ERROR_QUIET
)

if (CP2K_PATCH_CAN_APPLY EQUAL 0)
        message(STATUS "Applying CP2K benchmark patch: ${CP2K_PATCH_FILE}")
        execute_process(
                COMMAND "${GIT_EXECUTABLE}" apply "${CP2K_PATCH_FILE}"
                WORKING_DIRECTORY "${CP2K_SOURCE_DIR}"
                RESULT_VARIABLE CP2K_PATCH_APPLY_RESULT
        )
        if (NOT CP2K_PATCH_APPLY_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to apply CP2K patch: ${CP2K_PATCH_FILE}")
        endif ()
else ()
        execute_process(
                COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${CP2K_PATCH_FILE}"
                WORKING_DIRECTORY "${CP2K_SOURCE_DIR}"
                RESULT_VARIABLE CP2K_PATCH_ALREADY_APPLIED
                OUTPUT_QUIET
                ERROR_QUIET
        )
        if (CP2K_PATCH_ALREADY_APPLIED EQUAL 0)
                message(STATUS "CP2K benchmark patch is already applied.")
        else ()
                message(FATAL_ERROR
                        "CP2K benchmark patch cannot be applied cleanly and is not already applied. "
                        "Check the CP2K checkout state in ${CP2K_SOURCE_DIR}.")
        endif ()
endif ()

set(CP2K_SOURCE_FILES
        cp2k/src/base/machine_cpuid.c
        cp2k/src/common/cp_data_dir.c

        cp2k/src/base/kinds.F
        cp2k/src/hfx_types.F
        cp2k/src/base/machine.F
        cp2k/src/base/base_hooks.F
        cp2k/src/common/cp_files.F
        cp2k/src/hfxbase/hfx_compression_core_methods.F
        cp2k/src/hfx_compression_methods.F
)

add_library(cp2k_compression STATIC ${CP2K_SOURCE_FILES})
target_compile_options(cp2k_compression PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-ffree-form -ffree-line-length-none -g -march=native>)
target_compile_definitions(cp2k_compression PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:__SHORT_FILE__=\"__FILE__\">
        __DATA_DIR=""
)
if (OpenMP_C_FOUND)
        target_link_libraries(cp2k_compression PRIVATE OpenMP::OpenMP_C)
endif ()
if (OpenMP_Fortran_FOUND)
        target_link_libraries(cp2k_compression PRIVATE OpenMP::OpenMP_Fortran)
endif ()
set_target_properties(cp2k_compression PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran_modules/cp2k"
)

cmake_minimum_required(VERSION 3.24)
project(ollama-stable-diffusion C CXX)

# SD.cpp subproject: builds stable-diffusion.cpp as a shared library per
# backend, mirroring the llama/server CMake subproject conventions.
#
# Installed layout under lib/ollama/sdcpp/<backend>/:
#   libstable-diffusion.{so,dylib,dll}
#
# Shared-library visibility (OLLAMA_SDCPP_SHARED) keeps each ggml's internal
# symbols private to its shared object, avoiding conflicts with llama.cpp's
# vendored ggml when both are loaded in the same process.

if(NOT DEFINED OLLAMA_SOURCE_DIR OR "${OLLAMA_SOURCE_DIR}" STREQUAL "")
    set(OLLAMA_SOURCE_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Ollama repository root")
endif()
get_filename_component(OLLAMA_SOURCE_DIR "${OLLAMA_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

# Read pinned SD.cpp commit from version file (shared with Dockerfile)
file(READ "${OLLAMA_SOURCE_DIR}/SD_CPP_VERSION" SD_CPP_GIT_TAG)
string(STRIP "${SD_CPP_GIT_TAG}" SD_CPP_GIT_TAG)

# Ensure shared libraries can find their dependencies in the same directory.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_RPATH "@loader_path")
elseif(NOT WIN32)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

include(FetchContent)

# Allow local source override via environment variable.
if(DEFINED ENV{OLLAMA_SD_CPP_SOURCE})
    get_filename_component(_sdcpp_src "$ENV{OLLAMA_SD_CPP_SOURCE}" ABSOLUTE BASE_DIR "${OLLAMA_SOURCE_DIR}")
    set(FETCHCONTENT_SOURCE_DIR_STABLE_DIFFUSION_CPP "${_sdcpp_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local stable-diffusion.cpp source: ${_sdcpp_src}")
endif()

# Configure SD.cpp build options BEFORE FetchContent_MakeAvailable.
# Build stable-diffusion as a shared library with ggml+backends embedded
# statically inside it (SD_BUILD_SHARED_GGML_LIB=OFF). Combined with hidden
# default visibility, this keeps each ggml's internal symbols private to the
# libstable-diffusion shared object, avoiding conflicts with llama.cpp's
# vendored ggml when both are loaded in the same process (Phase 0.4).
#
# Backends are linked statically into the shared lib (no GGML_BACKEND_DL, which
# would emit separate ggml backend modules whose symbols escape the hidden-
# visibility boundary). The superbuild caller selects the backend via
# GGML_<BACKEND>=ON; GGML_CPU is ON by default in ggml.
set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "Build stable-diffusion as shared lib" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "Embed ggml statically into libstable-diffusion" FORCE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden CACHE STRING "" FORCE)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "" FORCE)
# GGML_BACKEND_DL and GGML_CPU_ALL_VARIANTS must be OFF so ggml backends are
# linked statically into the shared lib instead of emitted as separate modules.
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
# With a single statically-linked CPU backend, ggml's default GGML_NATIVE=ON
# would compile it with -march=native, baking the build host's ISA into the
# distributed payload (SIGILL on any other CPU). Force the portable x86-64
# baseline (SSE4.2/AVX/AVX2/FMA/F16C/BMI2; AVX512 stays off) instead.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# Container output support (WebM/animated-WebP) for video encoding.
set(USE_WEBP ON CACHE BOOL "Enable WebP support" FORCE)
set(USE_WEBM ON CACHE BOOL "Enable WebM video container support" FORCE)

FetchContent_Declare(
    stable_diffusion_cpp
    GIT_REPOSITORY "https://github.com/leejet/stable-diffusion.cpp.git"
    GIT_TAG ${SD_CPP_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stable_diffusion_cpp)

# --- Phase 0.4: ggml symbol isolation ---------------------------------------
# SD.cpp vendors its own ggml and links it statically into the shared
# libstable-diffusion. The static ggml objects are compiled with default
# (visible) visibility, so without intervention their symbols leak into the
# shared library's dynamic symbol table and clash with llama.cpp's vendored
# ggml when both are loaded in the same process.
#
# Two layers enforce isolation on ELF (Linux) and Mach-O (macOS) hosts:
#   1. Compile everything with -fvisibility=hidden so non-SD_API symbols are
#      hidden by default. SD_API functions are explicitly marked with
#      __attribute__((visibility("default"))) in the header, so they remain
#      exported. This is applied to ggml-base, ggml-cpu, and the SD lib.
#   2. (ELF only) A linker version script exports ONLY the SD_API surface,
#      guaranteeing no ggml_* or internal C++ symbols are visible regardless
#      of how upstream compiles its objects. This is the authoritative gate.
if(TARGET stable-diffusion)
    set(_sd_target stable-diffusion)
elseif(TARGET sd)
    set(_sd_target sd)
else()
    set(_sd_target)
endif()

if(_sd_target)
    # Layer 1: hidden compile visibility for the SD lib and its ggml deps.
    foreach(_vis_target "${_sd_target}" ggml-base ggml-cpu ggml)
        if(TARGET ${_vis_target})
            set_target_properties(${_vis_target} PROPERTIES
                CXX_VISIBILITY_PRESET hidden
                C_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON)
        endif()
    endforeach()

    # Layer 2: linker version script (ELF only) exporting only SD_API symbols.
    # The public C API symbols all start with sd_, new_sd_, free_sd_,
    # generate_image, generate_video, or convert.
    if(NOT WIN32 AND NOT APPLE)
        set(_sdcpp_version_script "${CMAKE_CURRENT_BINARY_DIR}/sdcpp.exports")
        file(WRITE "${_sdcpp_version_script}"
"SD_API_1.0 {\n"
"    global:\n"
"        sd_*;\n"
"        new_sd_ctx;\n"
"        free_sd_ctx;\n"
"        free_sd_images;\n"
"        free_sd_audio;\n"
"        generate_image;\n"
"        generate_video;\n"
"        convert;\n"
"        go*Trampoline;\n"
"    local:\n"
"        *;\n"
"};\n")
        set_target_properties(${_sd_target} PROPERTIES
            LINK_FLAGS "-Wl,--version-script,${_sdcpp_version_script}")
    endif()
endif()

set(OLLAMA_LIB_DIR "lib/ollama" CACHE STRING "Install destination for Ollama runtime payloads")
set(_base_dest "${OLLAMA_LIB_DIR}")
set(_sdcpp_dest "${_base_dest}/sdcpp/${OLLAMA_SDCPP_RUNNER_DIR}")

# Install the stable-diffusion shared library into the backend-specific dir.
if(TARGET stable-diffusion)
    install(TARGETS stable-diffusion
        RUNTIME DESTINATION "${_sdcpp_dest}" COMPONENT sdcpp OPTIONAL
        LIBRARY DESTINATION "${_sdcpp_dest}" COMPONENT sdcpp OPTIONAL)
elseif(TARGET sd)
    install(TARGETS sd
        RUNTIME DESTINATION "${_sdcpp_dest}" COMPONENT sdcpp OPTIONAL
        LIBRARY DESTINATION "${_sdcpp_dest}" COMPONENT sdcpp OPTIONAL)
endif()

# Install vendored header so the CGO bridge can find it at build time.
install(FILES "${OLLAMA_SOURCE_DIR}/x/sdcpp/include/stable-diffusion.h"
    DESTINATION "${_base_dest}/sdcpp/include" COMPONENT sdcpp OPTIONAL)

# Bundle ggml shared libraries built alongside SD.cpp into the backend dir.
install(CODE "
    set(_backend_dirs \"\${CMAKE_BINARY_DIR}/bin\")
    if(DEFINED CMAKE_INSTALL_CONFIG_NAME AND NOT CMAKE_INSTALL_CONFIG_NAME STREQUAL \"\")
        list(PREPEND _backend_dirs \"\${CMAKE_BINARY_DIR}/bin/\${CMAKE_INSTALL_CONFIG_NAME}\")
    endif()
    foreach(_dir IN LISTS _backend_dirs)
        file(GLOB _sdcpp_libs
            LIST_DIRECTORIES false
            \"\${_dir}/libstable-diffusion*\"
            \"\${_dir}/stable-diffusion*.dll\"
            \"\${_dir}/libggml*\"
            \"\${_dir}/ggml*.dll\"
        )
        list(APPEND _libs \${_sdcpp_libs})
    endforeach()
    foreach(_f \${_libs})
        file(INSTALL \${_f} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_sdcpp_dest}\")
    endforeach()
" COMPONENT sdcpp)

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
# Build as a shared library with hidden default visibility except the SD_API
# surface, so internal ggml symbols do not clash with llama.cpp's ggml.
set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "Build stable-diffusion as shared lib" FORCE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden CACHE STRING "" FORCE)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "" FORCE)

# Container output support (WebM/animated-WebP) for video encoding.
set(USE_WEBP ON CACHE BOOL "Enable WebP support" FORCE)
set(USE_WEBM ON CACHE BOOL "Enable WebM video container support" FORCE)

# Forward GGML backend selection from the superbuild caller. The caller sets
# GGML_<BACKEND>=ON via the OLLAMA_SDCPP_BACKENDS mapping before configuring.
if(NOT DEFINED GGML_CPU_ALL_VARIANTS)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    stable_diffusion_cpp
    GIT_REPOSITORY "https://github.com/leejet/stable-diffusion.cpp.git"
    GIT_TAG ${SD_CPP_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stable_diffusion_cpp)

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

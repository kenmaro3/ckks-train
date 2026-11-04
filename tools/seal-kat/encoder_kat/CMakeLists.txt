cmake_minimum_required(VERSION 3.20)
project(hekit_seal_encoder_kat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Microsoft SEAL 4.1.2 (MIT) via FetchContent, same settings as spikes/seal-2020.
# Offline: -DFETCHCONTENT_SOURCE_DIR_SEAL=<path to an existing SEAL v4.1.2 checkout>.
include(FetchContent)
set(SEAL_BUILD_DEPS ON CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(SEAL_USE_INTEL_HEXL OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_SEAL_C OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  SEAL
  GIT_REPOSITORY https://github.com/microsoft/SEAL.git
  GIT_TAG v4.1.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(SEAL)

add_executable(encoder_kat kat.cpp)
target_link_libraries(encoder_kat PRIVATE SEAL::seal)

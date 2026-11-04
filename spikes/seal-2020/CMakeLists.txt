cmake_minimum_required(VERSION 3.20)
project(seal2020_spike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# --- Microsoft SEAL 4.1 (MIT) via FetchContent; sources land in build/_deps (gitignored) ---
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
set(SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  SEAL
  GIT_REPOSITORY https://github.com/microsoft/SEAL.git
  GIT_TAG v4.1.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(SEAL)

add_executable(he_train src/he_train.cpp)
target_link_libraries(he_train PRIVATE SEAL::seal)

add_executable(seal_ntt_bench src/seal_ntt_bench.cpp)
target_link_libraries(seal_ntt_bench PRIVATE SEAL::seal)

enable_testing()
add_test(NAME he_train_check COMMAND he_train --check
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME he_train_check_alg4 COMMAND he_train --check --transpose alg4
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

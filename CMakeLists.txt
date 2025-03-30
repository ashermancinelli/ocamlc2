cmake_minimum_required(VERSION 3.12)
project(OcamlC2 VERSION 0.1.0 LANGUAGES C CXX)
cmake_policy(SET CMP0116 NEW)

# Add our custom cmake modules path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# Find OCaml compiler
find_package(OCaml REQUIRED)

set(CMAKE_GENERATOR Ninja)

# Global build settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ENABLE_SANITIZERS ON CACHE BOOL "Enable sanitizers")

set(TREESITTER_OCAML_DIR "/usr/local" CACHE PATH "Path to Tree-sitter OCaml parser install")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# enable warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wunused-variable -Wunused-value -Wunused-function -Werror")

include(ExternalProject)

if (ENABLE_SANITIZERS)
    include(Sanitizers)
endif()

if (NOT TREESITTER_OCAML_DIR)
    message(STATUS "Building Tree-sitter OCaml parser")
    include(AddTreeSitter)
else()
    message(STATUS "Tree-sitter OCaml parser pre-built directory: ${TREESITTER_OCAML_DIR}")
    include_directories(SYSTEM ${TREESITTER_OCAML_DIR}/include)
    find_file(TREESITTER_OCAML_HEADER tree-sitter-ocaml.h HINTS ${TREESITTER_OCAML_DIR}/include/tree_sitter)
    get_filename_component(TREESITTER_OCAML_HEADER_DIR ${TREESITTER_OCAML_HEADER} DIRECTORY)
    include_directories(SYSTEM ${TREESITTER_OCAML_HEADER_DIR})
    message(STATUS "Tree-sitter OCaml header directory: ${TREESITTER_OCAML_HEADER_DIR}")
    find_library(TREESITTER_OCAML_LIBRARY NAMES tree-sitter-ocaml HINTS ${TREESITTER_OCAML_DIR}/lib /usr/include /usr/local/include)
    link_libraries(${TREESITTER_OCAML_LIBRARY})
    find_library(TREESITTER_LIBRARY NAMES tree-sitter HINTS ${TREESITTER_OCAML_DIR}/lib /usr/include /usr/local/include)
    link_libraries(${TREESITTER_LIBRARY})
    message(STATUS "Tree-sitter OCaml libraries: ${TREESITTER_OCAML_LIBRARY} ${TREESITTER_LIBRARY}")
endif()

if(NOT MLIR_DIR AND NOT LLVM_DIR)
    set(MLIR_DIR "/usr/local/lib/cmake/mlir" CACHE PATH "Path to MLIR install")
    set(LLVM_DIR "/usr/local/lib/cmake/llvm" CACHE PATH "Path to LLVM install")
endif()

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add LLVM includes and definitions
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

# Disable RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

message(STATUS "Using FileCheck: ${LLVM_TOOLS_DIR}/FileCheck")

# Add MLIR includes and definitions
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(include)

add_subdirectory(tpl)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)

# Print configuration information
message(STATUS "OCaml compiler found: ${OCAML_COMPILER}")
message(STATUS "OCaml version: ${OCAML_VERSION}")

# Configure version header
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OcamlC2Config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/OcamlC2Config.h
) 

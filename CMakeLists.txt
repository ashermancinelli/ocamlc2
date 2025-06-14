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
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(BUILD_WITH_INSTALL_RPATH ON)
list(APPEND CMAKE_INSTALL_RPATH "/usr/local/lib")
list(APPEND BUILD_RPATH "/usr/local/lib")
list(APPEND INSTALL_RPATH "/usr/local/lib")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ENABLE_SANITIZERS ON CACHE BOOL "Enable sanitizers")
set(OCAMLRT_DEBUG ON CACHE BOOL "Enable OCamlRT debugging output")
set(ENABLE_COVERAGE OFF CACHE BOOL "Enable code coverage")

set(TREESITTER_OCAML_DIR "/usr/local" CACHE PATH "Path to Tree-sitter OCaml parser install")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(OCAMLC2_LINKER_FLAGS "" CACHE STRING "Linker flags for ocamlc2")

enable_testing()

# enable warnings
set(CMAKE_CXX_FLAGS 
  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wunused-variable -Wunused-value -Wunused-function -Werror -Wno-covered-switch-default -Wno-deprecated-declarations")

include(ExternalProject)
include(OCamlLinkerFlags)

if (ENABLE_SANITIZERS)
    include(Sanitizers)
endif()

if (ENABLE_COVERAGE)
    include(CodeCoverage)
    enable_coverage()
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
include_directories(${PROJECT_SOURCE_DIR}/tpl/cpp-tree-sitter/include)
add_definitions(${LLVM_DEFINITIONS})

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
    find_library(TREESITTER_OCAML_INTERFACE_LIBRARY NAMES tree-sitter-ocaml-interface HINTS ${TREESITTER_OCAML_DIR}/lib /usr/include /usr/local/include)
    link_libraries(${TREESITTER_OCAML_INTERFACE_LIBRARY})
    find_library(TREESITTER_LIBRARY NAMES tree-sitter HINTS ${TREESITTER_OCAML_DIR}/lib /usr/include /usr/local/include)
    link_libraries(${TREESITTER_LIBRARY})
    message(STATUS "Tree-sitter OCaml libraries: ${TREESITTER_OCAML_LIBRARY} ${TREESITTER_OCAML_INTERFACE_LIBRARY} ${TREESITTER_LIBRARY}")
endif()

find_program(LIT_COMMAND NAMES lit llvm-lit)
if (NOT LIT_COMMAND)
    message(FATAL_ERROR "lit or llvm-lit not found")
else()
    message(STATUS "Using lit: ${LIT_COMMAND}")
endif()

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

include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tpl/cpp-subprocess)

add_subdirectory(tpl)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(stdlib)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(MLIR_LIBS ${dialect_libs} ${conversion_libs} ${extension_libs} MLIRIR MLIRAsmParser MLIRParser MLIRSCFDialect MLIRFuncDialect MLIRLLVMDialect MLIRCallInterfaces MLIRCastInterfaces MLIRFunctionInterfaces MLIRTransforms MLIRArithToLLVM MLIRFuncToLLVM)

add_subdirectory(tools)
add_subdirectory(test)

# Print configuration information
message(STATUS "OCaml compiler found: ${OCAML_COMPILER}")
message(STATUS "OCaml version: ${OCAML_VERSION}")
message(STATUS "cppo executable: ${CPPO_EXECUTABLE}")

# Configure version header
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OCamlC2Config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/ocamlc2/OcamlC2Config.h
) 

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OCamlRTConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/ocamlc2/OCamlRTConfig.h
)

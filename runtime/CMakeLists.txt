set(RUNTIME_SOURCES
    ocamlrt.cpp
)

add_library(ocamlrt_static STATIC ${RUNTIME_SOURCES})
add_library(ocamlrt_static_thinlto STATIC ${RUNTIME_SOURCES})

target_compile_options(ocamlrt_static PRIVATE -O3)
target_compile_options(ocamlrt_static_thinlto PRIVATE -O3 -flto=thin)
target_compile_options(ocamlrt_static_thinlto PRIVATE -flto=thin)
target_link_options(ocamlrt_static_thinlto PRIVATE -flto=thin)
set_target_properties(ocamlrt_static ocamlrt_static_thinlto
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_library(ocamlrt INTERFACE)
add_dependencies(ocamlrt ocamlrt_static ocamlrt_static_thinlto)
add_custom_target(runtimes DEPENDS ocamlrt)

install(TARGETS ocamlrt_static ocamlrt_static_thinlto DESTINATION lib)

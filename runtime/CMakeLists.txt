set(RUNTIME_SOURCES
    ocamlrt.cpp
)

add_library(ocamlrt_static STATIC ${RUNTIME_SOURCES})
add_library(ocamlrt_static_thinlto STATIC ${RUNTIME_SOURCES})
# Set extra compile flags for specific targets
target_compile_options(ocamlrt_static PRIVATE -O3)
target_compile_options(ocamlrt_static_thinlto PRIVATE -O3 -flto=thin)
target_compile_options(ocamlrt_static_thinlto PRIVATE -flto=thin)
target_link_options(ocamlrt_static_thinlto PRIVATE -flto=thin)

add_library(ocamlrt INTERFACE)
add_dependencies(ocamlrt ocamlrt_static ocamlrt_static_thinlto)

install(TARGETS ocamlrt_static ocamlrt_static_thinlto DESTINATION lib)

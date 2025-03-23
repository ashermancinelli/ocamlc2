add_library(ocamlrt STATIC
    ocamlrt.cpp
)

install(TARGETS ocamlrt DESTINATION lib)

add_executable(ocamlc2-parse ocamlc2-parse.cpp)

target_link_libraries(ocamlc2-parse PRIVATE
    ${TREESITTER_OCAML_LIBRARY}
)

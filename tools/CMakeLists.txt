add_executable(ocamlc2-parse ocamlc2-parse.cpp)

target_link_libraries(ocamlc2-parse PRIVATE
    ${TREESITTER_OCAML_LIBRARY}
)
set_target_properties(ocamlc2-parse PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS ocamlc2-parse DESTINATION bin)

add_executable(ocamlc2-parse ocamlc2-parse.cpp)
target_link_libraries(ocamlc2-parse PRIVATE
    ${TREESITTER_OCAML_LIBRARY}
    MLIRIR
    LLVMSupport
    Parse
    Support
)
set_target_properties(ocamlc2-parse PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS ocamlc2-parse DESTINATION bin)

add_executable(ocamlc2-tomlir ocamlc2-tomlir.cpp)
target_link_libraries(ocamlc2-tomlir PRIVATE
    ${TREESITTER_OCAML_LIBRARY}
    MLIRParser
    MLIRSCFDialect
    MLIRTransforms
    MLIRArithToLLVM
    MLIRFuncToLLVM
    LLVMSupport
    Parse
    Support
)
set_target_properties(ocamlc2-tomlir PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS ocamlc2-tomlir DESTINATION bin)

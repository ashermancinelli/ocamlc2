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
    MLIROcaml
)
set_target_properties(ocamlc2-tomlir PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS ocamlc2-tomlir DESTINATION bin)

# Copy the 'c' script to binary directory
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/c
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/c ${CMAKE_BINARY_DIR}/bin/c
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/c
    COMMENT "Copying compiler driver script to bin directory"
)

# Add a custom target that depends on the script
add_custom_target(copy_c_script ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/c)

# Install the script
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/c
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

function(add_ocamlc2_tool name)
    message(STATUS "Adding tool: ${name}")
    cmake_parse_arguments(ARG "LINK_MLIR_LIBS" "" "" ${ARGN})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        LLVMSupport
        SemanticAnalysis
        CamlParse
        Support
        MLIRIR
        MLIRSCFToControlFlow
    )
    add_dependencies(${name} stdlib-interfaces)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(ARG_LINK_MLIR_LIBS)
        message(STATUS "Linking MLIR libs for tool ${name}")
        target_link_libraries(${name} PRIVATE ${MLIR_LIBS})
    endif()
    install(TARGETS ${name} DESTINATION bin)
endfunction()

# add_ocamlc2_tool(ocamlc2-link ocamlc2-link.cpp)
# add_ocamlc2_tool(ocamlc2-parse ocamlc2-parse.cpp)
# add_ocamlc2_tool(ocamlc2-tomlir ocamlc2-tomlir.cpp LINK_MLIR_LIBS)

# add_ocamlc2_tool(ocamlc2-parse2 ocamlc2-parse2.cpp)
add_ocamlc2_tool(p3 p3.cpp)
# add_ocamlc2_tool(ocamlc2-camlparse ocamlc2-camlparse.cpp)
# add_ocamlc2_tool(ocamlc2-tomlir2 ocamlc2-tomlir2.cpp LINK_MLIR_LIBS)
add_ocamlc2_tool(g3 g3.cpp LINK_MLIR_LIBS)
add_ocamlc2_tool(ocaml-opt ocaml-opt.cpp LINK_MLIR_LIBS)
target_link_libraries(ocaml-opt PRIVATE MLIROptLib MLIRTransforms)
add_ocamlc2_tool(ocaml-lsp-server ocaml-lsp-server.cpp LINK_MLIR_LIBS)
target_link_libraries(ocaml-lsp-server PRIVATE MLIRLspServerLib)

# Copy the 'c' script to binary directory
# add_custom_command(
#     OUTPUT ${CMAKE_BINARY_DIR}/bin/c
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/c ${CMAKE_BINARY_DIR}/bin/c
#     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/c
#     COMMENT "Copying compiler driver script to bin directory"
# )

# Add a custom target that depends on the script
# add_custom_target(copy_c_script ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/c)

# # Install the script
# install(
#     FILES ${CMAKE_CURRENT_SOURCE_DIR}/c
#     DESTINATION bin
#     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#                 GROUP_READ GROUP_EXECUTE
#                 WORLD_READ WORLD_EXECUTE
# )

# install(TARGETS ocamlc2-link ocamlc2-parse ocamlc2-tomlir DESTINATION bin)

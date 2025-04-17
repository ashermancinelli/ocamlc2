get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(MLIR_LIBS ${dialect_libs} ${conversion_libs} ${extension_libs} MLIRIR MLIRAsmParser MLIRParser MLIRSCFDialect MLIRFuncDialect MLIRLLVMDialect MLIRCallInterfaces MLIRCastInterfaces MLIRFunctionInterfaces MLIRTransforms MLIRArithToLLVM MLIRFuncToLLVM)

function(add_ocamlc2_tool name)
    message(STATUS "Adding tool: ${name}")
    cmake_parse_arguments(ARG "LINK_MLIR_LIBS" "" "" ${ARGN})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        LLVMSupport
        Parse
        CamlParse
        Support
        MLIRIR
    )
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(ARG_LINK_MLIR_LIBS)
        message(STATUS "Linking MLIR libs for tool ${name}")
        target_link_libraries(${name} PRIVATE ${MLIR_LIBS})
    endif()
    install(TARGETS ${name} DESTINATION bin)
endfunction()

add_ocamlc2_tool(ocamlc2-link ocamlc2-link.cpp)
add_ocamlc2_tool(ocamlc2-parse ocamlc2-parse.cpp)
add_ocamlc2_tool(ocamlc2-tomlir ocamlc2-tomlir.cpp LINK_MLIR_LIBS)

add_ocamlc2_tool(ocamlc2-parse2 ocamlc2-parse2.cpp)
add_ocamlc2_tool(p3 p3.cpp)
add_ocamlc2_tool(ocamlc2-camlparse ocamlc2-camlparse.cpp)
add_ocamlc2_tool(ocamlc2-tomlir2 ocamlc2-tomlir2.cpp LINK_MLIR_LIBS)

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

install(TARGETS ocamlc2-link ocamlc2-parse ocamlc2-tomlir DESTINATION bin)

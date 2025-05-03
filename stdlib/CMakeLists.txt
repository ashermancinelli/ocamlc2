set(INTERFACES
    array.mli
    string.mli
    bool.mli
    int.mli
    list.mli
    map.mli
    option.mli
    float.mli
    stdlib.mli
)
add_custom_target(stdlib-interfaces ALL
    COMMENT "Copying interfaces to distribution directory"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include/stdlib
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INTERFACES} ${CMAKE_BINARY_DIR}/include/stdlib
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(I DEPENDS stdlib-interfaces)

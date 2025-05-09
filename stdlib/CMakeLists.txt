set(INTERFACES
    array.mli
    bool.mli
    float.mli
    int.mli
    list.mli
    map.mli
    option.mli
    stdlib.mli
    string.mli
)
list(JOIN INTERFACES " " INTERFACES_STRING)
add_custom_target(stdlib-interfaces ALL
    COMMENT "Copying interfaces to distribution directory: ${INTERFACES_STRING}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include/stdlib
    COMMAND bash -c "install -m 644 ${INTERFACES_STRING} ${CMAKE_BINARY_DIR}/include/stdlib"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(I DEPENDS stdlib-interfaces)

set(INTERFACES
    array.mli
    string.mli
    bool.mli
    int.mli
    list.mli
    map.mli
    option.mli
    stdlib.mli
)
add_custom_target(install-dist-interfaces ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include/dist
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INTERFACES} ${CMAKE_BINARY_DIR}/include/dist
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

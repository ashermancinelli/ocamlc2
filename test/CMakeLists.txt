add_custom_target(check
    COMMAND ${LIT_COMMAND} -svv ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS all stdlib-interfaces
)

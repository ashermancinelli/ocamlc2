add_subdirectory(cpp-tree-sitter)
add_custom_target(check
    COMMAND ${LIT_COMMAND} -svv ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS all stdlib-interfaces
)

if (ENABLE_COVERAGE)
    # Find the main executable target
    # Use p3 as the target for coverage
    set(COVERAGE_EXECUTABLE $<TARGET_FILE:p3>)
    
    # Setup coverage target
    setup_coverage_target(
        NAME coverage
        EXECUTABLE ${COVERAGE_EXECUTABLE}
        DEPENDENCIES p3 stdlib-interfaces all
    )
endif()

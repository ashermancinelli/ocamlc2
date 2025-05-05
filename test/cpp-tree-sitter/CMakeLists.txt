add_executable(traversal traversal.cpp)
target_link_libraries(traversal PRIVATE SemanticAnalysis Support)
add_test(
    NAME traversal
    COMMAND bash -c "$<TARGET_FILE:traversal> | FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/traversal.cpp"
)

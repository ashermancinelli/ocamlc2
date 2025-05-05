
# Function to add a unit test that pipes output to FileCheck
function(add_unit_test test_name)
  # Create the test executable from the source file
  add_executable(${test_name} ${test_name}.cpp)
  
  # Link against necessary libraries
  target_link_libraries(${test_name} PRIVATE
    ${MLIR_LIBS}
    LLVMSupport
    SemanticAnalysis
    CamlParse
    Support
  )
  
  # Add the test that runs the executable and pipes to FileCheck
  add_test(
    NAME ${test_name}
    COMMAND
    bash -c "$<TARGET_FILE:${test_name}> | FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp"
  )
  
  # Set test properties
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "LLVM_DISABLE_CRASH_REPORT=1"
  )
endfunction()

add_unit_test(types)

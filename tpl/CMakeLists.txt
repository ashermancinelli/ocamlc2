set(BDWGC_EXTRA_C_FLAGS "" CACHE STRING "Extra C flags for bdwgc")
set(BDWGC_BUILD_TYPE "Release" CACHE STRING "Build type for bdwgc")
ExternalProject_Add(bdwgc
  GIT_REPOSITORY https://github.com/ivmai/bdwgc.git
  GIT_TAG master
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/bdwgc
  CMAKE_ARGS 
    -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
    -DCMAKE_BUILD_TYPE=${BDWGC_BUILD_TYPE}
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_C_FLAGS=${BDWGC_EXTRA_C_FLAGS}
  UPDATE_COMMAND ""
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/bdwgc
)
# Install the bdwgc libraries to the project's lib directory
ExternalProject_Add_Step(bdwgc install_libs
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_BINARY_DIR}/lib/ ${CMAKE_BINARY_DIR}/lib/
  DEPENDEES install
)

# Add a custom target to glob and copy the bdwgc libraries
add_custom_target(copy_bdwgc_libs ALL
  COMMAND ${CMAKE_COMMAND} -E echo "Copying bdwgc libraries into build directory"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_BINARY_DIR}/lib/libgc.* ${CMAKE_BINARY_DIR}/lib/ || true
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_BINARY_DIR}/lib/libcord.* ${CMAKE_BINARY_DIR}/lib/ || true
  DEPENDS bdwgc
)
install(FILES ${CMAKE_BINARY_DIR}/lib/libgc.* DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/lib/libcord.* DESTINATION lib)

add_subdirectory(cpp-tree-sitter)

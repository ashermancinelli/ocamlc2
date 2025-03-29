set(PUSH_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
add_mlir_dialect_library(MLIROcaml
    OcamlDialect.cpp

    DEPENDS
    OcamlIncGen
)
set(CMAKE_CXX_FLAGS "${PUSH_CMAKE_CXX_FLAGS}")

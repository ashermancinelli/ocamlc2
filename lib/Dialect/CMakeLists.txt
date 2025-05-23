# set(PUSH_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
add_mlir_dialect_library(MLIROcaml
    OcamlDialect.cpp
    OcamlTypeUtils.cpp
    BufferizeBoxes.cpp
    ConvertOCamlToLLVM.cpp
    LowerOCamlRuntime.cpp
    OcamlOpBuilder.cpp
    TypeInference.cpp
    Setup.cpp
    DEPENDS
    OcamlIncGen
    OcamlPassesIncGen
    
    LINK_LIBS PUBLIC
    LLVMCore
    LLVMSupport
    MLIRIR
    MLIRSCFToControlFlow
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRPass
    MLIRTransforms
)
# set(CMAKE_CXX_FLAGS "${PUSH_CMAKE_CXX_FLAGS}")

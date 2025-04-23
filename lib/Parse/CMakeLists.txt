add_library(Parse TSAdaptor.cpp TSUtil.cpp MLIRGen.cpp MLIRGen2.cpp MLIRGen3.cpp Runtime.cpp AST.cpp TSTypeSystem.cpp TypeSystem.cpp ASTPasses.cpp)
target_link_libraries(Parse PUBLIC LLVMSupport)

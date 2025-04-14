add_library(Parse TSAdaptor.cpp MLIRGen.cpp MLIRGen2.cpp Runtime.cpp AST.cpp TypeSystem.cpp ASTPasses.cpp)
target_link_libraries(Parse PUBLIC LLVMSupport)

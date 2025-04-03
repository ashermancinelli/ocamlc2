add_library(Parse TSAdaptor.cpp MLIRGen.cpp Runtime.cpp AST.cpp)
target_link_libraries(Parse PUBLIC LLVMSupport)

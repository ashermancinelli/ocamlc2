add_library(Parse TSAdaptor.cpp MLIRGen.cpp)
target_link_libraries(Parse PUBLIC LLVMSupport)

add_library(SemanticAnalysis
  TSAdaptor.cpp
  TSUtil.cpp
  MLIRGen3.cpp
  AST.cpp
  Unifier/Unifier.cpp
  Unifier/SourceUtilities.cpp
  Unifier/TypeUtilities.cpp
  Unifier/TypeSystem.cpp
  Unifier/Inference.cpp
  Unifier/Environment.cpp
)
target_link_libraries(SemanticAnalysis PUBLIC LLVMSupport Support)

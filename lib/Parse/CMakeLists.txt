add_library(SemanticAnalysis
  TSAdaptor.cpp
  TSUtil.cpp
  MLIRGen.cpp
  MLIRGen2.cpp
  MLIRGen3.cpp
  Runtime.cpp
  AST.cpp
  TypeSystem.cpp
  ASTPasses.cpp
  TSTypeSystem.cpp
  Unifier/Unifier.cpp
  Unifier/SourceUtilities.cpp
  Unifier/TypeUtilities.cpp
  Unifier/Inference.cpp
  Unifier/Environment.cpp
)
target_link_libraries(SemanticAnalysis PUBLIC LLVMSupport Support)

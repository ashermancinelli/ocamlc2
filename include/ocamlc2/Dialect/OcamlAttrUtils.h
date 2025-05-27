#pragma once
#include "OcamlDialect.h"

namespace mlir::ocaml {

llvm::StringRef getVariantCtorAttrName();
llvm::StringRef getRecursiveAttrName();
llvm::StringRef getUnresolvedFunctionAttrName();
llvm::StringRef getExternalFunctionAttrName();
llvm::StringRef getEnvironmentAttrName();
llvm::StringRef getEnvironmentIsForFunctionAttrName();
llvm::StringRef getOcamlAttributePrefix();
llvm::StringRef getOcamlFunctorAttrName();

}

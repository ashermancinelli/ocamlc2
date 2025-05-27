#include "ocamlc2/Dialect/OcamlAttrUtils.h"
#include "ocamlc2/Dialect/OcamlDialect.h"

llvm::StringRef mlir::ocaml::getVariantCtorAttrName() {
  return "ocaml.variant_ctor";
}

llvm::StringRef mlir::ocaml::getRecursiveAttrName() {
  return "ocaml.recursive";
}

llvm::StringRef mlir::ocaml::getUnresolvedFunctionAttrName() {
  return "ocaml.unresolved";
}

llvm::StringRef mlir::ocaml::getExternalFunctionAttrName() {
  return "ocaml.external";
}

llvm::StringRef mlir::ocaml::getEnvironmentAttrName() { return "ocaml.env"; }

llvm::StringRef mlir::ocaml::getEnvironmentIsForFunctionAttrName() {
  return "ocaml.env_for_function";
}

llvm::StringRef mlir::ocaml::getOcamlAttributePrefix() { return "ocaml."; }

llvm::StringRef mlir::ocaml::getOcamlFunctorAttrName() {
  return "ocaml.functor";
}

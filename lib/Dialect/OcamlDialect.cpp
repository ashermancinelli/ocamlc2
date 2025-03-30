#include "ocamlc2/Dialect/OcamlDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir::ocaml;
using namespace mlir;

#include "ocamlc2/Dialect/OcamlDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ocamlc2/Dialect/OcamlTypes.cpp.inc"

#define GET_OP_CLASSES
#include "ocamlc2/Dialect/OcamlOps.cpp.inc"

void OcamlDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ocamlc2/Dialect/OcamlTypes.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "ocamlc2/Dialect/OcamlOps.cpp.inc"
  >();
}

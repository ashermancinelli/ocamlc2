#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "ocamlc2/Dialect/OcamlDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ocamlc2/Dialect/OcamlTypes.h.inc"

#define GET_OP_CLASSES
#include "ocamlc2/Dialect/OcamlOps.h.inc"

namespace mlir::ocaml {
  inline bool isa_box_type(mlir::Type type) {
    return isa<BoxType, OpaqueBoxType, StringType, UnitType>(type);
  }
}

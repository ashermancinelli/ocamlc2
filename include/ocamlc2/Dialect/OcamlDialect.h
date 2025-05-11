#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ocamlc2/Dialect/OcamlDialect.h.inc"
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/PassManager.h>

namespace mlir::ocaml {
namespace detail {
struct VariantTypeStorage;
}
}

#define GET_TYPEDEF_CLASSES
#include "ocamlc2/Dialect/OcamlTypes.h.inc"

#define GET_OP_CLASSES
#include "ocamlc2/Dialect/OcamlOps.h.inc"

namespace mlir::ocaml {
inline bool isa_box_type(mlir::Type type) {
  return isa<BoxType, OpaqueBoxType, StringType, UnitType>(type);
}
llvm::StringRef getVariantCtorAttrName();
void setupRegistry(mlir::DialectRegistry &registry);
void setupContext(mlir::MLIRContext &context);
void setupDefaultPipeline(mlir::PassManager &pm);
void setupCodegenPipeline(mlir::PassManager &pm);
mlir::NamedAttribute getMatchCaseAttr(mlir::MLIRContext *context);
bool hasMatchCaseAttr(mlir::Operation *op);

}

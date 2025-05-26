#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "ocamlc2/Dialect/OcamlAttrUtils.h"
#include "ocamlc2/Dialect/OcamlTypeUtils.h"
#include "ocamlc2/Dialect/OcamlDialect.h.inc"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Pass/PassManager.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

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

void setupRegistry(mlir::DialectRegistry &registry);
void setupContext(mlir::MLIRContext &context);
void setupDefaultPipeline(mlir::PassManager &pm);
void setupCodegenPipeline(mlir::PassManager &pm);
mlir::NamedAttribute getMatchCaseAttr(mlir::MLIRContext *context);
bool hasMatchCaseAttr(mlir::Operation *op);

}

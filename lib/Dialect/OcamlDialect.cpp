#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectInterface.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/OpDefinition.h>
#include "mlir/IR/DialectImplementation.h"

using namespace mlir::ocaml;
using namespace mlir;

#include "ocamlc2/Dialect/OcamlDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ocamlc2/Dialect/OcamlTypes.cpp.inc"

#define GET_OP_CLASSES
#include "ocamlc2/Dialect/OcamlOps.cpp.inc"

OpFoldResult ConvertOp::fold(ConvertOp::FoldAdaptor adaptor) {
  auto input = getInput();
  if (getFromType() == getToType()) {
    return input;
  }
  if (auto def = input.getDefiningOp()) {
    if (auto definingConvert = mlir::dyn_cast<mlir::ocaml::ConvertOp>(def)) {
      if (definingConvert.getFromType() == getToType()) {
        return definingConvert.getInput();
      }
    }
  }
  return nullptr;
}

namespace mlir {
struct OcamlInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                        Type resultType,
                                        Location conversionLoc) const final {
    return builder.create<ConvertOp>(conversionLoc, resultType, input);
  }
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  void handleTerminator(Operation *op,
                        ValueRange valuesToRepl) const final {
    auto returnOp = cast<func::ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

} // namespace mlir


void OcamlDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ocamlc2/Dialect/OcamlTypes.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "ocamlc2/Dialect/OcamlOps.cpp.inc"
  >();
  addInterfaces<OcamlInlinerInterface>();
}

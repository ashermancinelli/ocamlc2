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

// `variant` `<` $name `is` $ctor `of` $type (`|` $ctor `of` $type)* `>`
mlir::Type VariantType::parse(mlir::AsmParser &parser) {
  std::string name;
  mlir::SmallVector<mlir::Type> elements;
  mlir::SmallVector<mlir::StringAttr> ctors;
  if (parser.parseLess())
    return {};
  if (parser.parseString(&name))
    return {};
  if (parser.parseKeyword("is"))
    return {};

  auto parseCtorAndType = [&] -> LogicalResult {
    std::string ctor;
    mlir::Type type;
    if (parser.parseString(&ctor))
      return mlir::failure();
    if (parser.parseKeyword("of"))
      return mlir::failure();
    if (parser.parseType(type))
      return mlir::failure();
    elements.push_back(type);
    ctors.push_back(mlir::StringAttr::get(parser.getContext(), ctor));
    return mlir::success();
  };

  if (failed(parseCtorAndType()))
    return {};

  while (parser.parseOptionalKeyword("|")) {
    if (failed(parseCtorAndType()))
      return {};
  }

  if (parser.parseGreater())
    return {};

  mlir::StringAttr nameAttr = mlir::StringAttr::get(parser.getContext(), name);
  return parser.getChecked<VariantType>(parser.getContext(), nameAttr, ctors, elements);
}

void VariantType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getName() << " is ";
  for (auto ctor : llvm::enumerate(getConstructors())) {
    printer << ctor.value() << " of " << getTypes()[ctor.index()];
    if (ctor.index() < getConstructors().size() - 1) {
      printer << " | ";
    }
  }
  printer << ">";
}

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

#include "ocamlc2/Dialect/OcamlDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include "ocamlc2/Dialect/OcamlTypeUtils.h"
#include "ocamlc2/Dialect/TypeDetail.h"
#include "llvm/ADT/TypeSwitch.h"
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/OpDefinition.h>
#include <optional>

#define DEBUG_TYPE "ocaml-dialect"
#include "ocamlc2/Support/Debug.h.inc"

using namespace mlir::ocaml;

#include "ocamlc2/Dialect/OcamlDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ocamlc2/Dialect/OcamlTypes.cpp.inc"

#define GET_OP_CLASSES
#include "ocamlc2/Dialect/OcamlOps.cpp.inc"

void mlir::ocaml::LoadOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ' << getInput();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType();
}

mlir::ParseResult mlir::ocaml::LoadOp::parse(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  if (parser.parseOperand(inputRawOperand))
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  mlir::Type type;
  if (parser.parseColonType(type))
    return mlir::failure();
  mlir::Value input;
  if (parser.resolveOperand(inputRawOperand, type, result.operands))
    return mlir::failure();
  result.addOperands({input});
  result.addTypes(type);
  return mlir::success();
}

void mlir::ocaml::StoreOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ' << getValue() << " to " << getInput() << " : "
          << getInput().getType();
  printer.printOptionalAttrDict((*this)->getAttrs());
}

mlir::ParseResult mlir::ocaml::StoreOp::parse(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  if (parser.parseOperand(valueRawOperand))
    return mlir::failure();
  if (parser.parseKeyword("to"))
    return mlir::failure();
  if (parser.parseOperand(inputRawOperand))
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  mlir::Type type;
  if (parser.parseColonType(type))
    return mlir::failure();
  mlir::Value value;
  mlir::Value input;
  if (parser.resolveOperand(valueRawOperand, value.getType(), result.operands))
    return mlir::failure();
  if (parser.resolveOperand(inputRawOperand, type, result.operands))
    return mlir::failure();
  result.addOperands({value, input});
  result.addTypes(type);
  return mlir::success();
}

mlir::LogicalResult mlir::ocaml::StoreOp::verify() {
  auto valueType = getValue().getType();
  auto inputType = getInput().getType();
  auto referenceType = mlir::cast<mlir::ocaml::ReferenceType>(inputType);
  if (!referenceType) {
    return emitError() << "input is not a reference type";
  }
  if (valueType != referenceType.getElementType()) {
    return emitError() << "value type " << valueType
                       << " does not match input type "
                       << referenceType.getElementType();
  }
  return mlir::success();
}

void mlir::ocaml::ClosureOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &result,
                                   mlir::func::FuncOp funcOp, mlir::Value env) {
  auto closureType = mlir::ocaml::ClosureType::get(builder.getContext(),
                                                   funcOp.getFunctionType());
  build(builder, result, closureType, funcOp.getSymName(), env);
}

void mlir::ocaml::CurryOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &result,
                                 mlir::Value closure, mlir::ValueRange args) {
  auto closureType = mlir::cast<mlir::ocaml::ClosureType>(closure.getType());
  auto functionType = closureType.getFunctionType();
  llvm::SmallVector<mlir::Value> converted;
  for (auto [i, arg] : llvm::enumerate(args)) {
    auto argType = functionType.getInput(i);
    DBGS("coercible? " << arg.getType() << " " << argType << "\n");
    assert(areTypesCoercible(arg.getType(), argType));
    if (arg.getType() != argType) {
      arg = builder.create<mlir::ocaml::ConvertOp>(arg.getLoc(), argType, arg);
    }
    converted.push_back(arg);
  }
  auto resultType = [&] -> mlir::Type {
    if (functionType.getNumInputs() == converted.size()) {
      return functionType.getResult(0);
    } else {
      SmallVector<mlir::Type> newFunctionInputTypes;
      llvm::append_range(
          newFunctionInputTypes,
          llvm::drop_begin(functionType.getInputs(), converted.size()));
      auto newFunctionType =
          mlir::FunctionType::get(builder.getContext(), newFunctionInputTypes,
                                  functionType.getResults());
      return mlir::ocaml::ClosureType::get(builder.getContext(),
                                           newFunctionType);
    }
  }();
  build(builder, result, resultType, closure, args);
}

void mlir::ocaml::ListConsOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ' << getValue() << " :: " << getList() << " : " << getType();
}

mlir::ParseResult mlir::ocaml::ListConsOp::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  mlir::OpAsmParser::UnresolvedOperand listRawOperand{};
  if (parser.parseOperand(valueRawOperand))
    return mlir::failure();
  if (parser.parseKeyword("::"))
    return mlir::failure();
  if (parser.parseOperand(listRawOperand))
    return mlir::failure();
  mlir::Type type;
  if (parser.parseColonType(type))
    return mlir::failure();
  mlir::Value value;
  mlir::Value list;
  if (parser.resolveOperand(valueRawOperand, value.getType(), result.operands))
    return mlir::failure();
  if (parser.resolveOperand(listRawOperand, list.getType(), result.operands))
    return mlir::failure();
  result.addOperands({value, list});
  result.addTypes(type);
  return mlir::success();
}

void mlir::ocaml::ClosureType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getFunctionType() << ">";
}

mlir::Type mlir::ocaml::ClosureType::parse(mlir::AsmParser &parser) {
  mlir::FunctionType functionType;
  if (parser.parseLess())
    return {};
  if (parser.parseType(functionType))
    return {};
  if (parser.parseGreater())
    return {};
  return parser.getChecked<ClosureType>(parser.getContext(), functionType);
}

void mlir::ocaml::CallOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &result,
                                mlir::Value closure, mlir::ValueRange args) {
  TRACE();
  DBGS("closure: " << closure.getType() << " with " << args.size()
                   << " args\n");
  auto closureType = mlir::cast<mlir::ocaml::ClosureType>(closure.getType());
  auto functionType = closureType.getFunctionType();
  assert(args.size() == functionType.getNumInputs());
  auto inputs = functionType.getInputs();
  llvm::SmallVector<mlir::Value> converted;
  for (auto [arg, argType] : llvm::zip_equal(args, inputs)) {
    DBGS("coercible? " << arg.getType() << " " << argType << "\n");
    assert(areTypesCoercible(arg.getType(), argType));
    if (arg.getType() != argType) {
      arg = builder.create<mlir::ocaml::ConvertOp>(arg.getLoc(), argType, arg);
    }
    converted.push_back(arg);
  }
  auto resultType = functionType.getResult(0);
  build(builder, result, resultType, closure, args, {}, {});
}

mlir::ParseResult mlir::ocaml::GlobalOp::parse(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  mlir::StringAttr name;
  mlir::Type type;
  if (parser.parseSymbolName(name, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return mlir::failure();
  if (parser.parseColonType(type))
    return mlir::failure();
  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));
  return mlir::success();
}

void mlir::ocaml::GlobalOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getSymName() << " : " << getType();
}

void mlir::ocaml::GlobalOp::build(mlir::OpBuilder &builder,
                                  mlir::OperationState &result,
                                  llvm::StringRef name, mlir::Type type,
                                  llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getSymrefAttrName(result.name),
                      mlir::SymbolRefAttr::get(builder.getContext(), name));
}

mlir::LogicalResult mlir::ocaml::ArrayFromElementsOp::verify() {
  return mlir::success();
}

mlir::NamedAttribute mlir::ocaml::getMatchCaseAttr(mlir::MLIRContext *context) {
  auto name = mlir::ocaml::getOcamlAttributePrefix() + "match_case";
  return mlir::NamedAttribute(mlir::StringAttr::get(context, name),
                              mlir::UnitAttr::get(context));
}

// `variant` `<` $name `is` $ctor `of` $type (`|` $ctor `of` $type)* `>`
mlir::Type mlir::ocaml::VariantType::parse(mlir::AsmParser &parser) {
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
    if (failed(parser.parseString(&ctor)))
      return mlir::failure();
    if (succeeded(parser.parseOptionalKeyword("of"))) {
      if (failed(parser.parseType(type)))
        return mlir::failure();
    } else {
      type = UnitType::get(parser.getContext());
    }
    elements.push_back(type);
    ctors.push_back(mlir::StringAttr::get(parser.getContext(), ctor));
    return mlir::success();
  };

  if (failed(parseCtorAndType()))
    return {};

  while (succeeded(parser.parseOptionalKeyword("or"))) {
    if (failed(parseCtorAndType()))
      return {};
  }

  if (parser.parseGreater())
    return {};

  mlir::StringAttr nameAttr = mlir::StringAttr::get(parser.getContext(), name);
  return parser.getChecked<VariantType>(parser.getContext(), nameAttr, ctors,
                                        elements);
}

void VariantType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getName() << " is ";
  for (auto iter : llvm::enumerate(llvm::zip(getConstructors(), getTypes()))) {
    auto [ctor, type] = iter.value();
    printer << ctor;
    if (type != UnitType::get(getContext())) {
      printer << " of " << type;
    }
    if (iter.index() < getConstructors().size() - 1) {
      printer << " or ";
    }
  }
  printer << ">";
}

mlir::FailureOr<std::pair<unsigned, mlir::Type>>
mlir::ocaml::VariantType::typeForConstructor(llvm::StringRef name,
                                             VariantType type) {
  for (auto iter :
       llvm::enumerate(llvm::zip(type.getConstructors(), type.getTypes()))) {
    auto [ctor, type] = iter.value();
    if (ctor == name) {
      return {std::make_pair(iter.index(), type)};
    }
  }
  return mlir::failure();
}

mlir::OpFoldResult
mlir::ocaml::ConvertOp::fold(ConvertOp::FoldAdaptor adaptor) {
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
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
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

namespace mlir::ocaml {} // namespace mlir::ocaml

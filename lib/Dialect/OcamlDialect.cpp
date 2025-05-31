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

namespace mlir::ocaml {
namespace detail {
struct ModuleTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::StringRef;

  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.str());
  }

  bool operator==(const KeyTy &key) const { return key == getName(); }

  static ModuleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    auto *storage = allocator.allocate<ModuleTypeStorage>();
    return new (storage) ModuleTypeStorage{key};
  }

  llvm::StringRef getName() const { return name; }

  void setTypeList(llvm::ArrayRef<ModuleType::TypePair> list) { types = list; }
  llvm::ArrayRef<ModuleType::TypePair> getTypeList() const { return types; }
  void addType(llvm::StringRef ident, mlir::Type type) {
    DBGS("adding type " << ident << " : " << type << " to module type " << name << "\n");
    types.emplace_back(ident, type);
  }

  FailureOr<mlir::Type> getType(llvm::StringRef ident) const {
    for (auto [i, type] : llvm::enumerate(types)) {
      if (type.first == ident) {
        return type.second;
      }
    }
    return failure();
  }
  FailureOr<mlir::Type> getType(unsigned index) const {
    if (index >= types.size()) {
      return failure();
    }
    return types[index].second;
  }
  unsigned getNumFields() const { return types.size(); }
  FailureOr<unsigned> getFieldIndex(llvm::StringRef ident) const {
    for (auto [i, type] : llvm::enumerate(types)) {
      if (type.first == ident) {
        return i;
      }
    }
    return failure();
  }
  void finalize() {
    DBGS("finalizing module type " << name << " with " << types.size() << " types\n");
    assert(!finalized);
    finalized = true;
  }
  bool isFinalized() const { return finalized; }
  void finalize(llvm::ArrayRef<ModuleType::TypePair> typeList) {
    assert(!finalized);
    finalized = true;
    setTypeList(typeList);
  }

protected:
  std::string name;
  bool finalized;
  std::vector<ModuleType::TypePair> types;

private:
  ModuleTypeStorage() = delete;
  explicit ModuleTypeStorage(llvm::StringRef name)
      : name{name}, finalized{false} {}
};
}
}

namespace {
static llvm::SmallPtrSet<detail::ModuleTypeStorage const *, 4> moduleTypeVisited;
}

void mlir::ocaml::ModuleOp::build(mlir::OpBuilder &builder,
                                  mlir::OperationState &result,
                                  llvm::StringRef name) {
  auto moduleType = mlir::ocaml::ModuleType::get(builder.getContext(), name);
  build(builder, result, moduleType, name);
}

void mlir::ocaml::ModuleType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printString(getName());
  if (!moduleTypeVisited.count(uniqueKey())) {
    printer << ", {";
    moduleTypeVisited.insert(uniqueKey());
    auto length = getTypeList().size();
    for (auto [i, type] : llvm::enumerate(getTypeList())) {
      printer.printString(StringRef(type.first));
      printer << " : ";
      printer.printType(type.second);
      if (i < length - 1) {
        printer << ", ";
      }
    }
    printer << "}";
    moduleTypeVisited.erase(uniqueKey());
  }
  printer << ">";
}

mlir::Type mlir::ocaml::ModuleType::parse(mlir::AsmParser &parser) {
  std::string name;
  ModuleType::TypeList typeList;
  if (parser.parseLess())
    return {};
  if (parser.parseString(&name))
    return {};
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseLBrace())
      return {};
    while (true) {
      std::string field;
      mlir::Type fldTy;
      if (parser.parseString(&field) || parser.parseColon() ||
          parser.parseType(fldTy)) {
        parser.emitError(parser.getNameLoc(), "expected type list");
        return {};
      }
      typeList.emplace_back(field, fldTy);
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRBrace())
      return {};
  }
  if (parser.parseGreater())
    return {};
  return parser.getChecked<ModuleType>(parser.getContext(), name);
}

llvm::LogicalResult mlir::ocaml::ModuleType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef name) {
  if (name.size() == 0)
    return emitError() << "record types must have a name";
  return mlir::success();
}

llvm::StringRef mlir::ocaml::ModuleType::getName() const {
  return getImpl()->getName();
}

mlir::ocaml::ModuleType::TypeList mlir::ocaml::ModuleType::getTypeList() const {
  return getImpl()->getTypeList();
}

bool mlir::ocaml::ModuleType::isFinalized() const { return getImpl()->isFinalized(); }

mlir::ocaml::detail::ModuleTypeStorage const *mlir::ocaml::ModuleType::uniqueKey() const {
  return getImpl();
}

void mlir::ocaml::ModuleType::addType(llvm::StringRef ident, mlir::Type type) {
  getImpl()->addType(ident, type);
}

llvm::FailureOr<mlir::Type> mlir::ocaml::ModuleType::getType(llvm::StringRef ident) const {
  return getImpl()->getType(ident);
}

llvm::FailureOr<mlir::Type> mlir::ocaml::ModuleType::getType(unsigned index) const {
  return getImpl()->getType(index);
}

void mlir::ocaml::ModuleType::finalize() {
  getImpl()->finalize();
}

void mlir::ocaml::ModuleType::finalize(llvm::ArrayRef<ModuleType::TypePair> typeList) {
  getImpl()->finalize(typeList);
}

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

mlir::StringAttr mlir::ocaml::EnvOp::getFor() const {
  return mlir::cast<mlir::StringAttr>((*this)->getAttr(mlir::ocaml::getEnvironmentIsForFunctionAttrName()));
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

  while (succeeded(parser.parseOptionalVerticalBar())) {
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
      printer << " | ";
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

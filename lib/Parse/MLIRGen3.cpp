#include "ocamlc2/Parse/MLIRGen3.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlTypeUtils.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/VisitorHelper.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <utility>
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/MonadHelpers.h"
#define DEBUG_TYPE "MLIRGen3.cpp"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;
using InsertionGuard = mlir::ocaml::OcamlOpBuilder::InsertionGuard;

struct ocamlc2::Scope {
  Scope(MLIRGen3 &gen) : gen(gen), variableScope(gen.variables) {}
  MLIRGen3 &gen;
  VariableScope variableScope;

};

using mlir::failed;
using mlir::failure;
using mlir::success;
using mlir::succeeded;
using mlir::LogicalResult;

static StringArena stringArena;

#define SSWRAP(x)                                                              \
  [&] {                                                                        \
    std::string str;                                                           \
    llvm::raw_string_ostream ss(str);                                          \
    ss << x;                                                                   \
    return ss.str();                                                           \
  }()

template <typename T>
mlir::FailureOr<T> mustBe(mlir::FailureOr<std::variant<T, mlir::func::FuncOp>> result) {
  if (failed(result)) {
    return failure();
  }
  if (std::holds_alternative<T>(*result)) {
    return std::get<T>(*result);
  }
  return failure();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetBindingValueDefinition(const Node patternNode, const Node bodyNode) {
  TRACE();
  auto patternType = unifierType(patternNode);
  auto maybeBodyValue = gen(bodyNode);
  if (failed(maybeBodyValue)) {
    return failure();
  }
  auto bodyValue = *maybeBodyValue;
  if (patternType == unifier.getUnitType()) {
    return bodyValue;
  }
  return declareVariable(patternNode, bodyValue, loc(patternNode));
}

mlir::FailureOr<Callee>
MLIRGen3::genFunctionBody(llvm::StringRef name, mlir::FunctionType funType,
                          mlir::Location location,
                          llvm::ArrayRef<Node> parameters, Node bodyNode) {
  DBGS("Generating function: " << name << "\n");
  InsertionGuard guard(builder);
  VariableScope scope(variables);
  mlir::ocaml::ClosureEnvValue env =
      builder.createEnv(location, getUniqueName((name + "env").str()));
  builder.setInsertionPointToStart(module->getBody());
  auto function =
      builder.create<mlir::func::FuncOp>(location, name, funType);
  function.setPrivate();
  function->setAttr("env", env.getFor());
  auto *bodyBlock = function.addEntryBlock();
  builder.setInsertionPointToEnd(bodyBlock);
  auto it = function.getArguments().begin();
  auto argRange = llvm::make_range(it, function.getArguments().end());
  for (auto [blockArg, parameter] : llvm::zip(argRange, parameters)) {
    auto parameterNode = parameter.getChildByFieldName("pattern");
    if (failed(declareVariable(parameterNode, blockArg, loc(parameterNode)))) {
      return failure();
    }
  }
  return gen(bodyNode) | and_then([&](auto body) -> FailureOr<Callee> {
    builder.create<mlir::func::ReturnOp>(location, body);
    return {function};
  });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetBinding(const Node node) {
  TRACE();
  const bool isRecursive = isLetBindingRecursive(node.getCursor());
  auto patternNode = node.getChildByFieldName("pattern");
  const auto patternType = unifierType(patternNode);
  auto bodyNode = node.getChildByFieldName("body");
  if (patternType == unifier.getUnitType()) {
    return gen(bodyNode);
  }
  if (isRecursive) {
    DBGS("recursive\n");
  }
  auto parameters = getNamedChildren(node, {"parameter"});
  if (parameters.empty()) {
    return genLetBindingValueDefinition(patternNode, bodyNode);
  }
  auto maybeFunctionType = mlirType(patternNode);
  if (failed(maybeFunctionType)) {
    return failure();
  }
  auto functionType = mlir::dyn_cast<mlir::FunctionType>(*maybeFunctionType);
  if (not functionType) {
    return error(node) << "Expected function type for let binding";
  }
  auto nameString = getText(patternNode);
  return genFunctionBody(nameString, functionType, loc(node), parameters,
                         bodyNode) |
             and_then([&](auto funcOp) -> mlir::FailureOr<mlir::Value> {
               return builder.createUnit(loc(node));
             });
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirTypeFromBasicTypeOperator(llvm::StringRef name) {
  if (name == "int") {
    return builder.emboxType(builder.getI64Type());
  } else if (name == "float") {
    return builder.emboxType(builder.getF64Type());
  } else if (name == "bool") {
    return builder.emboxType(builder.getI1Type());
  } else if (name == "unit") {
    return builder.getUnitType();
  } else if (name == "string") {
    return builder.getStringType();
  } else if (name == "_") {
    return builder.getOBoxType();
  }
  return failure();
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirFunctionType(const Node node) {
  auto *type = unifierType(node);
  return mlirFunctionType(type, loc(node));
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirFunctionType(ocamlc2::TypeExpr *type, mlir::Location loc) {
  TRACE();
  if (type == nullptr) {
    return error(loc) << "Type for node was not inferred during type inference";
  }
  if (const auto *to = llvm::dyn_cast<ocamlc2::FunctionOperator>(type)) {
    DBGS("Function type: " << *to << '\n');
    const auto typeOperatorArgs = to->getArgs();
    auto args = llvm::drop_end(typeOperatorArgs);
    auto argTypes = llvm::map_to_vector(args, [this, loc](auto *arg) { return mlirType(arg, loc); });
    if (llvm::any_of(argTypes, failed)) {
      return failure();
    }
    SmallVector<mlir::Type> successfulArgTypes;
    auto mappedTypes = llvm::map_to_vector(argTypes, [](auto t) { return *t; });
    llvm::append_range(successfulArgTypes, mappedTypes);
    auto returnType = *mlirType(typeOperatorArgs.back(), loc);
    return mlir::FunctionType::get(builder.getContext(), successfulArgTypes, {returnType});
  }
  return error(loc) << "Could get MLIR type from unified function type: "
                     << SSWRAP(*type);
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirVariantCtorType(ocamlc2::CtorOperator *ctor, mlir::Location loc) {
  TRACE();
  assert(ctor);
  assert(ctor->getArgs().size() == 2 &&
         "Variant ctor can only have 2 arguments, if more it must be a tuple, "
         "if less it's either unit or nullary ctor");
  auto toArgs = ctor->getArgs();
  return mlirType(toArgs.back(), loc) | and_then([&](auto returnType) -> mlir::FailureOr<mlir::Type> {
           auto argType = mlirType(toArgs.front(), loc);
           return argType | and_then([&](auto argType) -> mlir::FailureOr<mlir::Type> {
             return mlir::FunctionType::get(builder.getContext(), {argType}, {returnType});
           });
         });
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(ocamlc2::VariantOperator *type, mlir::Location loc) {
  auto name = type->getName();
  auto constructors = type->getConstructors();
  llvm::SmallVector<mlir::Type> constructorTypes;
  llvm::SmallVector<llvm::StringRef> constructorNames;
  auto visitor = Overload{
    [&](const std::pair<llvm::StringRef, ocamlc2::FunctionOperator *> &constructor) -> LogicalResult {
      auto *constructorTypeOperator = constructor.second;
      auto args = constructorTypeOperator->getArgs();
      if (args.size() != 2) {
        DBGS("constructor: " << *constructorTypeOperator << "\n");
        return error(loc) << "Expected 1 argument for constructor: " << name;
      }
      auto constructorType = mlirType(args.front(), loc);
      if (failed(constructorType)) {
        return failure();
      }
      constructorNames.push_back(constructor.first);
      constructorTypes.emplace_back(std::move(*constructorType));
      return success();
    },
    [&](const llvm::StringRef &constructor) {
      constructorNames.push_back(constructor);
      constructorTypes.emplace_back(builder.getUnitType());
      return success();
    }
  };
  for (auto constructor : constructors) {
    if (failed(std::visit(visitor, constructor))) {
      return failure();
    }
  }
  auto nameAttr = builder.getStringAttr(name);
  auto constructorNameAttrs = builder.createStringAttrVector(constructorNames);
  auto variantType = mlir::ocaml::VariantType::get(
      builder.getContext(), nameAttr, constructorNameAttrs, constructorTypes);
  return variantType;
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(ocamlc2::TypeExpr *type, mlir::Location loc) {
  DBGS("mlirType: " << *type << "\n");
  type = Unifier::pruneEverything(type);
  if (auto *ctor = llvm::dyn_cast<ocamlc2::CtorOperator>(type)) {
    TRACE();
    return mlirVariantCtorType(ctor, loc);
  } else if (auto *funcOperator = llvm::dyn_cast<ocamlc2::FunctionOperator>(type)) {
    TRACE();
    return mlirFunctionType(funcOperator, loc);
  } else if (auto *vo = llvm::dyn_cast<ocamlc2::VariantOperator>(type)) {
    TRACE();
    return mlirType(vo, loc);
  } else if (auto *to = llvm::dyn_cast<ocamlc2::TupleOperator>(type)) {
    TRACE();
    auto args = to->getArgs();
    llvm::SmallVector<mlir::Type> argTypes;
    for (auto *arg : args) {
      auto argType = mlirType(arg, loc);
      if (failed(argType)) {
        return failure();
      }
      argTypes.push_back(*argType);
    }
    return mlir::TupleType::get(builder.getContext(), argTypes);
  } else if (const auto *to = llvm::dyn_cast<ocamlc2::TypeOperator>(type)) {
    TRACE();
    const auto args = to->getArgs();
    if (args.empty()) {
      auto mlirType = mlirTypeFromBasicTypeOperator(to->getName());
      if (failed(mlirType)) {
        return error(loc) << "Unknown basic type operator: " << SSWRAP(*type);
      }
      return mlirType;
    }
    auto name = to->getName();
    if (name == "array") {
      auto elementType = mlirType(args.front(), loc);
      if (failed(elementType)) {
        return failure();
      }
      return builder.getArrayType(*elementType);
    }
    return error(loc) << "Unknown type operator: " << SSWRAP(*type);
  } else if (const auto *tv = llvm::dyn_cast<ocamlc2::TypeVariable>(type)) {
    TRACE();
    if (tv->instantiated()) {
      return mlirType(tv->instance, loc);
    }
    return builder.getOBoxType();
  }
  return error(loc) << "Unknown type: " << SSWRAP(*type);
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(const Node node) {
  auto *type = unifierType(node);
  if (type == nullptr) {
    return error(node) << "Type for node was not inferred at unification-time";
  }
  if (auto convertedType = typeExprToMlirType.lookup(type)) {
    return convertedType;
  }
  auto convertedType = mlirType(type, loc(node));
  if (failed(convertedType)) {
    return failure();
  }
  typeExprToMlirType[type] = *convertedType;
  return convertedType;
}

mlir::FailureOr<Callee> MLIRGen3::genConstructorPath(const Node node) {
  TRACE();
  InsertionGuard guard(builder);
  auto constructorName = getText(node);
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  auto type = *maybeType;
  DBGS("type: " << type << "\n");

  // ------------------------------------------------------------------
  // 1. Nullary constructor (value) – the MLIR type is already VariantType.
  // ------------------------------------------------------------------
  if (auto variantType = mlir::dyn_cast<mlir::ocaml::VariantType>(type)) {
    // Check for an existing zero-arg constructor function for this tag.
    if (auto sym = module->lookupSymbol<mlir::func::FuncOp>(constructorName)) {
      // Emit a call to the existing function – produces the variant value.
      auto callVal = builder.createCall(loc(node), sym, mlir::ValueRange{});
      return {callVal};
    }

    // Otherwise create the helper function.
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module->getBody());
    auto funcTypeZero = builder.getFunctionType({}, variantType);
    auto func = builder.create<mlir::func::FuncOp>(loc(node), constructorName, funcTypeZero);
    func.setPrivate();
    func->setAttrs({builder.createVariantCtorAttr()});

    // Build the body: construct the variant value and return it.
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Determine the index of this constructor within the variant.
    auto ctorInfo = mlir::ocaml::VariantType::typeForConstructor(constructorName, variantType);
    if (failed(ctorInfo)) {
      return error(node) << "Unknown constructor: " << constructorName << " in type " << type;
    }
    auto [ctorIndex, /*ctorPayloadType*/ _] = *ctorInfo;

    auto indexConst = builder.createConstant(loc(node), builder.getI64Type(), ctorIndex);
    mlir::Value variantValue = builder.create<mlir::ocaml::IntrinsicOp>(
        loc(node), variantType, "variant_ctor_empty", mlir::ValueRange{indexConst});

    builder.create<mlir::func::ReturnOp>(loc(node), variantValue);
    builder.restoreInsertionPoint(ip);

    // Now that the function exists, emit a call in the original position.
    auto callVal = builder.createCall(loc(node), func, mlir::ValueRange{});
    return {callVal};
  }

  // ------------------------------------------------------------------
  // 2. Constructor with payload – treat the path as a function value.
  // ------------------------------------------------------------------
  auto funcType = llvm::dyn_cast<mlir::FunctionType>(type);
  if (!funcType) {
    return error(node) << "Expected function type if not a variant type directly";
  }
  if (auto sym = module->lookupSymbol<mlir::func::FuncOp>(constructorName)) {
    return {sym};
  }

  builder.setInsertionPointToStart(module->getBody());
  auto func =
      builder.create<mlir::func::FuncOp>(loc(node), constructorName, funcType);
  func->setAttrs({builder.createVariantCtorAttr()});
  func.setPrivate();
  func.addEntryBlock();
  builder.setInsertionPointToEnd(&func.front());
  DBGS("func: " << func << "\n");
  auto opaqueVariantType = funcType.getResult(0);
  auto variantType =
      llvm::dyn_cast<mlir::ocaml::VariantType>(opaqueVariantType);
  if (!variantType) {
    return error(node) << "Expected variant type";
  }
  auto ctorInfo = mlir::ocaml::VariantType::typeForConstructor(constructorName,
                                                               variantType);
  if (failed(ctorInfo)) {
    return error(node) << "Unknown constructor: " << constructorName
                       << " in type " << type;
  }
  auto [ctorIndex, ctorPayloadType] = *ctorInfo;
  if (mlir::isa<mlir::ocaml::UnitType>(ctorPayloadType)) {
    auto indexConst =
        builder.createConstant(loc(node), builder.getI64Type(), ctorIndex);
    mlir::Value variant = builder.create<mlir::ocaml::IntrinsicOp>(
        loc(node), variantType, "variant_ctor_empty",
        mlir::ValueRange{indexConst});
    return {variant};
  }

  auto &body = func.getFunctionBody();
  builder.setInsertionPointToEnd(&body.front());
  auto indexConst =
      builder.createConstant(loc(node), builder.getI64Type(), ctorIndex);
  SmallVector<mlir::Value> args{indexConst, func.getArgument(0)};
  auto constructedValue = builder.create<mlir::ocaml::IntrinsicOp>(
      loc(node), variantType, "designate_variant", args);
  builder.create<mlir::func::ReturnOp>(loc(node),
                                       mlir::ValueRange{constructedValue});
  DBGS("func: " << func << "\n");
  return {func};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genMatchExpression(const Node node) {
  TRACE();
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  auto returnType = *maybeType;
  auto scrutineeNode = node.getChildByFieldName("expression");
  auto maybeScrutineeValue = gen(scrutineeNode);
  if (failed(maybeScrutineeValue)) {
    return failure();
  }
  auto scrutineeValue = *maybeScrutineeValue;
  auto caseNodes = getNamedChildren(node, {"match_case"});
  return [&] -> mlir::FailureOr<mlir::Value> {
    InsertionGuard guard(builder);
    auto regionOp = builder.create<mlir::scf::ExecuteRegionOp>(loc(node), returnType);
    auto &region = regionOp.getRegion();
    auto &returnBlock = region.emplaceBlock();
    returnBlock.addArgument(returnType, loc(node));
    builder.setInsertionPointToEnd(&returnBlock);
    builder.create<mlir::scf::YieldOp>(loc(node), mlir::ValueRange{returnBlock.getArgument(0)});
    auto *matchFailureBlock = builder.createBlock(&returnBlock);
    auto falseOp = builder.create<mlir::arith::ConstantOp>(loc(node), builder.getI1Type(), builder.getBoolAttr(false));
    builder.create<mlir::cf::AssertOp>(loc(node), falseOp, "No match found");
    auto failureReturn = builder.createConvert(loc(node), returnType, falseOp);
    builder.create<mlir::cf::BranchOp>(loc(node), &returnBlock, mlir::ValueRange{failureReturn});
    for (auto caseNode : caseNodes) {
      auto *patternBlock = builder.createBlock(matchFailureBlock);
      auto *bodyBlock = builder.createBlock(matchFailureBlock);
      builder.setInsertionPointToEnd(patternBlock);
      auto patternNode = caseNode.getChildByFieldName("pattern");
      auto bodyNode = caseNode.getChildByFieldName("body");
      auto maybePatternValue = gen(patternNode);
      if (failed(maybePatternValue)) {
        return failure();
      }
      auto patternValue = *maybePatternValue;
      auto matchedPattern = builder.createPatternMatch(
          loc(caseNode), scrutineeValue, patternValue);
      builder.create<mlir::cf::CondBranchOp>(loc(caseNode), matchedPattern, bodyBlock, matchFailureBlock);
      builder.setInsertionPointToEnd(bodyBlock);
      auto bodyValue = gen(bodyNode);
      builder.create<mlir::cf::BranchOp>(loc(bodyNode), &returnBlock,
                                         mlir::ValueRange{*bodyValue});
      matchFailureBlock = patternBlock;
    }
    return regionOp.getResult(0);
  }();
}

mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(const Node node) {
  auto str = getText(node);
  DBGS("Getting variable: " << str << '\n');
  return getVariable(str, loc(node));
}

FailureOr<std::tuple<bool, mlir::func::FuncOp, mlir::func::FuncOp>>
MLIRGen3::valueIsFreeInCurrentContext(mlir::Value value) {
  TRACE();
  auto func = [&] -> mlir::func::FuncOp {
    auto *op = value.getDefiningOp();
    if (op) {
      DBGS("Op\n");
      return op->getParentOfType<mlir::func::FuncOp>();
    }
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (arg) {
      DBGS("Arg\n");
      auto parent = arg.getOwner()->getParent()->getParentOfType<mlir::func::FuncOp>();
      if (parent) {
        DBGS("Func: " << parent << "\n");
        return parent;
      }
      return nullptr;
    }
    DBGS("No func\n");
    return nullptr;
  }();
  if (!func) {
    DBGS("No func\n");
    return failure();
  }
  auto *currentBlock = builder.getInsertionBlock();
  auto currentFunc = currentBlock->getParent()->getParentOfType<mlir::func::FuncOp>();
  if (currentFunc == func) {
    DBGS("Current region\n");
    return {{false, currentFunc, func}};
  }
  DBGS("Not current region\n");
  return {{true, currentFunc, func}};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genGlobalForFreeVariable(
    mlir::Value value, llvm::StringRef name, mlir::func::FuncOp currentFunc,
    mlir::func::FuncOp definingFunc, mlir::Location loc) {
  DBGS(name << " is used in " << currentFunc.getSymName() << " captured from "
            << definingFunc.getSymName() << "\n");
  return findEnvForFunction(currentFunc) |
         and_then([&](mlir::Value env) -> mlir::LogicalResult {
           InsertionGuard guard(builder);
           builder.setInsertionPointAfterValue(env);
           builder.createEnvCapture(loc, env, name, value);
           return success();
         }) | and_then([&]() -> mlir::FailureOr<mlir::Value> {
           InsertionGuard guard(builder);
           builder.setInsertionPointToStart(&currentFunc.getFunctionBody().front());
           auto envArg = builder.create<mlir::ocaml::EnvGetCurrentOp>(loc);
           builder.setInsertionPointAfterValue(envArg);
           auto loadCapturedArg = builder.createEnvGet(loc, value.getType(), envArg, name);
           return {loadCapturedArg};
         }) | and_then([&] (mlir::Value loadedValue) -> FailureOr<mlir::Value> {
           return declareVariable(name, loadedValue, loc);
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(llvm::StringRef name, mlir::Location loc) {
  if (auto value = variables.lookup(name)) {
    auto isFree = valueIsFreeInCurrentContext(value);
    if (failed(isFree)) {
      return failure();
    }
    auto [isFreeV, currentFunc, definingFunc] = *isFree;
    if (isFreeV) {
      return genGlobalForFreeVariable(value, name, currentFunc, definingFunc, loc);
    }
    return value;
  }
  return failure();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genForExpression(const Node node) {
  TRACE();
  const auto children = getNamedChildren(node);
  const auto inductionVariableNode = children[0];
  const auto startNode = children[1];
  const auto endNode = children[2];
  const auto bodyNode = children[3];
  const auto upOrDownToNode = node.getChild(4);
  const auto stepSize = upOrDownToNode.getType() == "to"       ? 1
                        : upOrDownToNode.getType() == "downto" ? -1
                                                           : 0;
  if (stepSize == 0) {
    return error(node) << "Could not infer step size";
  }
  auto startValue = gen(startNode);
  auto endValue = gen(endNode);
  if (failed(startValue) || failed(endValue)) {
    return error(node) << "Failed to generate bounds for `for` expression";
  }
  mlir::Value start = builder.createUnbox(loc(startNode), *startValue);
  mlir::Value end = builder.createUnbox(loc(endNode), *endValue);
  mlir::Value step = builder.createConstant(loc(upOrDownToNode), builder.getI64Type(),
                                     stepSize);
  auto forOp = builder.create<mlir::scf::ForOp>(loc(node), start, end, step);
  auto bodyBlock = forOp.getBody();
  VariableScope scope(variables);
  builder.setInsertionPointToStart(bodyBlock);
  auto inductionVariable = builder.createEmbox(loc(inductionVariableNode), forOp.getInductionVar());
  auto res = declareVariable(inductionVariableNode, inductionVariable,
                             loc(inductionVariableNode));
  if (failed(res)) {
    return res;
  }
  auto bodyValue = gen(bodyNode);
  if (failed(bodyValue)) {
    return error(node) << "Failed to generate body for `for` expression";
  }
  return builder.createUnit(loc(node));
}

mlir::FailureOr<mlir::Value> MLIRGen3::genCompilationUnit(const Node node) {
  TRACE();
  VariableScope scope(variables);
  builder.setInsertionPointToStart(module->getBody());
  auto mainFunc = builder.create<mlir::func::FuncOp>(
      loc(node), "main", builder.getFunctionType({}, builder.getI32Type()));
  mainFunc.setPrivate();
  auto &body = mainFunc.getFunctionBody();
  builder.setInsertionPointToEnd(&body.emplaceBlock());
  {
    InsertionGuard guard(builder);
    for (auto child : getNamedChildren(node)) {
      if (child.getType() == "comment") {
        continue;
      }
      if (mlir::failed(gen(child))) {
        return mlir::failure();
      }
    }
  }
  // TODO: figure out return value
  auto zero = builder.createConstant(mainFunc.getLoc(), builder.getI32Type(), 0);
  builder.create<mlir::func::ReturnOp>(mainFunc.getLoc(), zero);
  return mlir::Value();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genNumber(const Node node) {
  TRACE();
  auto str = getText(node);
  auto type = mlirType(node);
  if (failed(type)) {
    return failure();
  }
  long long int intValue;
  double doubleValue;
  if (not str.getAsInteger(10, intValue)) {
    auto constant = builder.createConstant(loc(node), builder.getI64Type(), intValue);
    return builder.createEmbox(loc(node), constant);
  } else if (not str.getAsDouble(doubleValue)) {
    auto constant = builder.createConstant(loc(node), builder.getF64Type(), doubleValue);
    return builder.createEmbox(loc(node), constant);
  }
  return error(node) << "Failed to parse number: " << str;
}

mlir::FailureOr<Callee> MLIRGen3::genCallee(const Node node) {
  TRACE();
  auto str = unifier.getTextSaved(node);
  auto value = getVariable(str, loc(node));
  return value | and_then([&](mlir::Value value) -> mlir::FailureOr<Callee> {
           if (mlir::ocaml::Closure{value}) {
             return Callee{value};
           }
           return error(node) << "Expected function for callee: " << str;
         }) |
         or_else([&]() -> mlir::FailureOr<Callee> {
           auto callee = module->lookupSymbol<mlir::func::FuncOp>(str);
           if (callee) {
             return Callee{callee};
           }
           return failure();
         }) |
         or_else([&]() -> mlir::FailureOr<Callee> {
           auto type = mlirType(node);
           if (failed(type)) {
             return failure();
           }
           auto functionType = mlir::dyn_cast<mlir::FunctionType>(*type);
           if (not functionType) {
             return error(node) << "Failed to get callee: " << str;
           }
           InsertionGuard guard(builder);
           builder.setInsertionPointToStart(module->getBody());
           auto callee = builder.create<mlir::func::FuncOp>(loc(node), str, functionType);
           callee.setPrivate();
           return Callee{callee};
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetExpression(const Node node) {
  TRACE();
  VariableScope scope(variables);
  InsertionGuard guard(builder);
  auto definitions = getNamedChildren(node, {"value_definition"});
  auto body = node.getChildByFieldName("body");
  auto maybeBodyType = mlirType(body);
  if (failed(maybeBodyType)) {
    return failure();
  }
  auto bodyType = *maybeBodyType;
  auto letExpressionRegion = builder.create<mlir::scf::ExecuteRegionOp>(loc(node), bodyType);
  auto &bodyBlock = letExpressionRegion.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(&bodyBlock);
  for (auto definition : definitions) {
    if (failed(gen(definition))) {
      return failure();
    }
  }
  auto maybeBodyValue = gen(body);
  if (failed(maybeBodyValue)) {
    return failure();
  }

  if (!mlir::isa<mlir::ocaml::UnitType>(bodyType)) {
    auto bodyValue = *maybeBodyValue;
    builder.create<mlir::scf::YieldOp>(loc(node), mlir::ValueRange{bodyValue});
  } else {
    builder.create<mlir::scf::YieldOp>(
        loc(node), mlir::ValueRange{builder.createUnit(loc(node))});
  }
  return letExpressionRegion.getResult(0);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genInfixExpression(const Node node) {
  TRACE();
  const auto children = getNamedChildren(node);
  const auto lhs = children[0];
  const auto rhs = children[2];
  const auto op = children[1];
  auto l = loc(node);
  return genCallee(op) |
         and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
           return gen(lhs) |
                  and_then([&](auto lhsValue) -> mlir::FailureOr<mlir::Value> {
                    return gen(rhs) |
                           and_then([&](auto rhsValue) {
                             // Let genApplication add the env operand automatically
                             return genApplication(l, callee, {lhsValue, rhsValue});
                           });
                  });
         });
}

mlir::FailureOr<mlir::ocaml::ClosureEnvValue> MLIRGen3::findEnvForFunction(mlir::func::FuncOp funcOp) {
  auto envAttr = funcOp->getAttr("env");
  if (not envAttr) {
    return failure();
  }
  FailureOr<mlir::ocaml::ClosureEnvValue> env=failure();
  getModule().walk([&](mlir::ocaml::EnvOp envOp) {
    if (mlir::ocaml::ClosureEnvValue{envOp}.getFor() == envAttr) {
      env = success(envOp.getResult());
    }
  });
  return env;
}

mlir::FailureOr<mlir::Value> MLIRGen3::genApplication(mlir::Location loc, Callee callee, llvm::ArrayRef<mlir::Value> args) {
  return std::visit(
      Overload{
          [&](mlir::func::FuncOp funcOp) -> mlir::FailureOr<mlir::Value> {
            return builder.createCall(loc, funcOp, args);
          },
          [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
            return nyi(loc, "application of value");
          }},
      callee);
}

mlir::FailureOr<mlir::Value>
MLIRGen3::genApplicationExpression(const Node node) {
  TRACE();
  const auto children = getNamedChildren(node);
  const auto callee = children[0];
  DBGS("callee: " << getText(callee) << '\n');
  auto args = llvm::drop_begin(children);
  return genCallee(callee) |
         and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
           const auto maybeArgs = llvm::to_vector(llvm::map_range(
               args, [this](const Node &arg) { return gen(arg); }));
           if (llvm::any_of(maybeArgs, failed)) {
             return error(node) << "Failed to generate arguments";
           }
           auto argValues = llvm::map_to_vector(
               maybeArgs, [](auto value) { return *value; });
           return genApplication(loc(node), callee, argValues);
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genArrayExpression(const Node node) {
  TRACE();
  auto arrayType = mlirType(node);
  if (failed(arrayType)) {
    return failure();
  }
  auto elementNodes = getNamedChildren(node);
  auto maybeElementTypes = llvm::to_vector(llvm::map_range(
      elementNodes, [this](const Node &arg) { return mlirType(arg); }));
  if (llvm::any_of(maybeElementTypes, failed)) {
    return error(node) << "Failed to generate element types";
  }
  auto elements = llvm::map_to_vector(
      elementNodes, [this](const Node &arg) { return gen(arg); });
  if (llvm::any_of(elements, failed)) {
    return error(node) << "Failed to generate elements";
  }
  auto elementValues =
      llvm::map_to_vector(elements, [](auto value) { return *value; });
  DBGS("array of type " << *arrayType << " with " << elementValues.size()
                        << " elements\n");
  auto array =
      builder.createArrayFromElements(loc(node), *arrayType, elementValues);
  assert(array.getType() == *arrayType);
  return array;
}

mlir::FailureOr<mlir::Value> MLIRGen3::genFunExpression(const Node node) {
  TRACE();
  auto parameters = getNamedChildren(node, {"parameter"});
  UU auto body = node.getChildByFieldName("body");
  return mlirFunctionType(node) |
         and_then([&](auto funType) -> mlir::FailureOr<mlir::FunctionType> {
           if (auto ft = mlir::dyn_cast<mlir::FunctionType>(funType)) {
             return ft;
           }
           return error(node)
                  << "Expected function type for fun expression: " << funType;
         }) |
         and_then([&](auto funType) {
           auto anonName = getUniqueName("funexpr");
           return genFunctionBody(anonName, funType, loc(node), parameters,
                                  body);
         }) |
         and_then([&](Callee callee) -> mlir::FailureOr<mlir::Value> {
           return std::visit(
               Overload{[&](mlir::func::FuncOp funcOp)
                            -> mlir::FailureOr<mlir::Value> {
                          auto constantFuncOp =
                              builder.create<mlir::func::ConstantOp>(
                                  loc(node), funcOp.getFunctionType(),
                                  funcOp.getSymName());
                          auto boxedType =
                              builder.emboxType(constantFuncOp.getType());
                          return builder.createConvert(loc(node), boxedType,
                                                       constantFuncOp);
                        },
                        [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
                          return nyi(loc(node), "application of value");
                        }},
               callee);
         });
}

mlir::FailureOr<mlir::Value> MLIRGen3::genArrayGetExpression(const Node node) {
  TRACE();
  auto arrayNode = node.getChildByFieldName("array");
  auto indexNode = node.getChildByFieldName("index");
  auto arrayValue = gen(arrayNode);
  if (failed(arrayValue)) {
    return error(node) << "Failed to generate array";
  }
  auto indexValue = gen(indexNode);
  if (failed(indexValue)) {
    return error(node) << "Failed to generate index";
  }
  return builder.createArrayGet(loc(node), *arrayValue, *indexValue);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genIfExpression(const Node node) {
  TRACE();
  InsertionGuard guard(builder);
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  std::optional<mlir::Type> type = *maybeType;
  auto conditionNode = node.getChildByFieldName("condition");
  auto thenNodes = getNamedChildren(node, {"then_clause"});
  assert(thenNodes.size() == 1);
  auto thenNode = thenNodes[0];
  auto elseNodes = getNamedChildren(node, {"else_clause"});
  auto elseNode =
      elseNodes.empty() ? std::nullopt : std::optional<Node>(elseNodes[0]);
  assert(elseNodes.size() <= 1);
  if (elseNodes.empty()) {
    assert(llvm::isa<mlir::ocaml::UnitType>(*type));
  }
  if (llvm::isa<mlir::ocaml::UnitType>(*type)) {
    type = std::nullopt;
  }
  auto maybeConditionValue = gen(conditionNode);
  if (failed(maybeConditionValue)) {
    return failure();
  }
  auto conditionValue = builder.createConvert(
      loc(conditionNode), builder.getI1Type(), *maybeConditionValue);
  SmallVector<mlir::Type> resultType;
  if (type) {
    resultType.push_back(*type);
  }

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc(node), resultType, conditionValue, true, elseNode.has_value());

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto thenValue = gen(thenNode);
  if (failed(thenValue)) {
    return failure();
  }

  if (!resultType.empty()) {
    builder.create<mlir::scf::YieldOp>(loc(node), mlir::ValueRange{*thenValue});
  }

  if (elseNode) {
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    auto elseValue = gen(elseNode.value());
    if (failed(elseValue)) {
      return failure();
    }
    builder.create<mlir::scf::YieldOp>(loc(node), mlir::ValueRange{*elseValue});
  }

  return {resultType.empty() ? builder.createUnit(loc(node))
                             : ifOp.getResult(0)};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genString(const Node node) {
  TRACE();
  auto str = getText(node);
  return builder.createString(loc(node), str);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genSequenceExpression(const Node node) {
  TRACE();
  auto children = getNamedChildren(node);
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  mlir::Value result;
  for (auto child : children) {
    auto maybeValue = gen(child);
    if (failed(maybeValue)) {
      return failure();
    }
    result = *maybeValue;
  }
  return result;
}

mlir::FailureOr<mlir::Value> MLIRGen3::genValuePattern(const Node node) {
  TRACE();
  auto maybeType = mlirType(node);
  if (failed(maybeType)) {
    return failure();
  }
  auto type = *maybeType;
  if (mlir::isa<mlir::ocaml::UnitType>(type)) {
    return builder.createUnit(loc(node));
  }

  auto placeholder = builder.createPatternVariable(loc(node), type);

  if (failed(declareVariable(node, placeholder, loc(node)))) {
    return failure();
  }

  return {placeholder};
}

mlir::FailureOr<mlir::Value> MLIRGen3::genConstructorPattern(const Node node) {
  TRACE();
  auto ctor = node.getNamedChild(0);
  auto maybeConstructorPath = genConstructorPath(ctor);
  if (failed(maybeConstructorPath)) {
    return failure();
  }
  auto constructorPath = *maybeConstructorPath;
  if (std::holds_alternative<mlir::Value>(constructorPath)) {
    return std::get<mlir::Value>(constructorPath);
  }
  auto constructorFunc = std::get<mlir::func::FuncOp>(constructorPath);
  auto constructorArgNodes =
      SmallVector<Node>(llvm::drop_begin(getNamedChildren(node)));
  auto maybeConstructorArgs = llvm::to_vector(llvm::map_range(
      constructorArgNodes, [this](const Node &arg) { return gen(arg); }));
  if (llvm::any_of(maybeConstructorArgs, failed)) {
    return failure();
  }
  auto constructorArgs = llvm::map_to_vector(maybeConstructorArgs,
                                             [](auto value) { return *value; });
  auto constructorCall =
      builder.createCall(loc(node), constructorFunc, constructorArgs);
  return constructorCall;
}

llvm::StringRef MLIRGen3::getText(const Node node) {
  return unifier.getTextSaved(node);
}

mlir::FailureOr<mlir::Value>
MLIRGen3::declareVariable(Node node, mlir::Value value, mlir::Location loc) {
  auto str = getText(node);
  if (node.getType() == "typed_pattern") {
    node = node.getChildByFieldName("pattern");
    str = getText(node);
  }
  DBGS("declaring variable: " << str << ' ' << node.getType() << "\n");
  return declareVariable(str, value, loc);
}

mlir::FailureOr<mlir::Value> MLIRGen3::declareVariable(llvm::StringRef name,
                                                       mlir::Value value,
                                                       mlir::Location loc) {
  auto savedName = stringArena.save(name);
  DBGS("declaring '" << savedName << "' of type " << value.getType() << "\n");
  // if (variables.count(savedName)) {
  //   return mlir::emitError(loc)
  //       << "Variable '" << name << "' already declared";
  // }
  variables.insert(savedName, value);
  return value;
}

mlir::FailureOr<mlir::Value> MLIRGen3::gen(const Node node) {
  TRACE();
  DBG(unifier.show(node.getCursor()));
  auto type = node.getType();
  InsertionGuard guard(builder);
  static constexpr std::string_view passthroughTypes[] = {
      "parenthesized_expression",
      "then_clause",
      "else_clause",
      "do_clause",
      "value_definition",
      "expression_item",
      "parenthesized_type",
  };
  static constexpr std::string_view ignoreTypes[] = {
      "comment",
      "type_definition",
  };
  if (llvm::is_contained(passthroughTypes, type)) {
    return gen(node.getNamedChild(0));
  } else if (type == "compilation_unit") {
    return genCompilationUnit(node);
  } else if (type == "let_binding") {
    return genLetBinding(node);
  } else if (llvm::is_contained(ignoreTypes, type)) {
    return mlir::Value();
  } else if (type == "number") {
    return genNumber(node);
  } else if (type == "for_expression") {
    return genForExpression(node);
  } else if (type == "application_expression") {
    return genApplicationExpression(node);
  } else if (type == "value_path") {
    return getVariable(node);
  } else if (type == "infix_expression") {
    return genInfixExpression(node);
  } else if (type == "unit") {
    return builder.createUnit(loc(node));
  } else if (type == "let_expression") {
    return genLetExpression(node);
  } else if (type == "value_name") {
    return getVariable(node);
  } else if (type == "match_expression") {
    return genMatchExpression(node);
  } else if (type == "constructor_path") {
    return genConstructorPath(node) |
           and_then([&](auto callee) -> mlir::FailureOr<mlir::Value> {
             return std::visit(
                 Overload{
                     [&](mlir::func::FuncOp funcOp)
                         -> mlir::FailureOr<mlir::Value> {
                       return {builder.create<mlir::func::ConstantOp>(
                           loc(node), funcOp.getFunctionType(),
                           funcOp.getSymName())};
                     },
                     [&](mlir::Value value) -> mlir::FailureOr<mlir::Value> {
                       return value;
                     }},
                 callee);
           });
  } else if (type == "constructor_pattern") {
    return genConstructorPattern(node);
  } else if (type == "value_pattern") {
    return genValuePattern(node);
  } else if (type == "sequence_expression") {
    return genSequenceExpression(node);
  } else if (type == "string") {
    return genString(node);
  } else if (type == "array_expression") {
    return genArrayExpression(node);
  } else if (type == "if_expression") {
    return genIfExpression(node);
  } else if (type == "array_get_expression") {
    return genArrayGetExpression(node);
  } else if (type == "fun_expression") {
    return genFunExpression(node);
  }
  return error(node) << "NYI: " << type << " (" << __LINE__ << ')';
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen3::gen() {
  TRACE();
  module = builder.create<mlir::ModuleOp>(loc(root), unifier.getLastModule());
  builder.setInsertionPointToStart(module->getBody());
  auto res = gen(root);
  if (mlir::failed(res)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    llvm::errs() << *module << "\n";
    return mlir::failure();
  }

  if (mlir::failed(module->verify())) {
    DBGS("Failed to verify MLIR for compilation unit\n");
    llvm::errs() << *module << "\n";
    return mlir::failure();
  }

  return std::move(module);
}

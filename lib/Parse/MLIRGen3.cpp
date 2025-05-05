#include "ocamlc2/Parse/MLIRGen3.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/VisitorHelper.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
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
#include <mlir/IR/Verifier.h>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/Utils.h"
#define DEBUG_TYPE "MLIRGen3.cpp"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;
using InsertionGuard = mlir::ocaml::OcamlOpBuilder::InsertionGuard;

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
    return nyi(node);
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
  mlir::func::FuncOp function;
  {
    DBGS("Generating function: " << nameString << "\n");
    InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module->getBody());
    function =
        builder.create<mlir::func::FuncOp>(loc(node), nameString, functionType);
    function.setPrivate();
    auto *bodyBlock = function.addEntryBlock();
    builder.setInsertionPointToEnd(bodyBlock);
    for (auto [blockArg, parameter] : llvm::zip(function.getArguments(), parameters)) {
      if (failed(declareVariable(parameter, blockArg, loc(parameter)))) {
        return failure();
      }
    }
    auto body = gen(bodyNode);
    if (failed(body)) {
      return failure();
    }
    builder.create<mlir::func::ReturnOp>(loc(node), *body);
  }
  return mlir::Value(builder.create<mlir::func::ConstantOp>(loc(node), function.getFunctionType(), function.getName()));
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
    auto successfulArgTypes = llvm::map_to_vector(argTypes, [](auto t) { return *t; });
    auto returnType = *mlirType(typeOperatorArgs.back(), loc);
    return mlir::FunctionType::get(builder.getContext(), successfulArgTypes, {returnType});
  }
  return error(loc) << "Could get MLIR type from unified function type: "
                     << SSWRAP(*type);
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

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(ocamlc2::TypeExpr *type,
                                               mlir::Location loc) {
  DBGS("mlirType: " << *type << "\n");
  type = Unifier::prune(type);
  if (auto *funcOperator = llvm::dyn_cast<ocamlc2::FunctionOperator>(type)) {
    return mlirFunctionType(funcOperator, loc);
  } else if (auto *vo = llvm::dyn_cast<ocamlc2::VariantOperator>(type)) {
    return mlirType(vo, loc);
  } else if (const auto *to = llvm::dyn_cast<ocamlc2::TypeOperator>(type)) {
    const auto args = to->getArgs();
    if (args.empty()) {
      auto mlirType = mlirTypeFromBasicTypeOperator(to->getName());
      if (failed(mlirType)) {
        return error(loc) << "Unknown basic type operator: " << SSWRAP(*type);
      }
      return mlirType;
    }
    return error(loc) << "Unknown type operator: " << SSWRAP(*type);
  } else if (const auto *tv = llvm::dyn_cast<ocamlc2::TypeVariable>(type)) {
    if (tv->instantiated()) {
      return mlirType(tv->instance, loc);
    }
    return error(loc) << "Uninstantiated type variable: " << SSWRAP(*type);
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


mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(const Node node) {
  auto str = getText(node);
  DBGS("Getting variable: " << str << '\n');
  return getVariable(str, loc(node));
}

mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(llvm::StringRef name, mlir::Location loc) {
  if (auto value = variables.lookup(name)) {
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

mlir::FailureOr<mlir::func::FuncOp> MLIRGen3::genCallee(const Node node) {
  TRACE();
  auto str = unifier.getTextSaved(node);
  auto callee = module->lookupSymbol<mlir::func::FuncOp>(str);
  if (!callee) {
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
    return callee;
  }
  return callee;
}

mlir::FailureOr<mlir::Value> MLIRGen3::genLetExpression(const Node node) {
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
  auto callee = genCallee(op);
  if (failed(callee)) {
    return error(node) << "Failed to get callee: " << op.getType();
  }
  auto lhsValue = gen(lhs);
  auto rhsValue = gen(rhs);
  if (failed(lhsValue) || failed(rhsValue)) {
    return error(node) << "Failed to generate lhs or rhs";
  }
  // llvm::dbgs() << *callee << "\n" << *lhsValue << "\n" << *rhsValue << "\n";
  return builder.createCall(loc(node), *callee, mlir::ValueRange{*lhsValue, *rhsValue});
}

mlir::FailureOr<mlir::Value> MLIRGen3::genApplicationExpression(const Node node) {
  const auto children = getNamedChildren(node);
  const auto callee = children[0];
  auto calleeFunc = genCallee(callee);
  if (mlir::failed(calleeFunc)) {
    return failure();
  }
  auto args = llvm::drop_begin(children);
  const auto maybeArgs = llvm::to_vector(llvm::map_range(
      args, [this](const Node &arg) { return gen(arg); }));
  if (llvm::any_of(maybeArgs, failed)) {
    return failure();
  }
  auto argValues = llvm::map_to_vector(maybeArgs, [](auto value) { return *value; });
  auto callOp = builder.createCall(loc(node), *calleeFunc, argValues);
  return callOp;
}

mlir::FailureOr<mlir::Value> MLIRGen3::gen(const Node node) {
  TRACE();
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
  }
  return error(node) << "NYI: " << type;
}

llvm::StringRef MLIRGen3::getText(const Node node) {
  return unifier.getTextSaved(node);
}

mlir::FailureOr<mlir::Value> MLIRGen3::declareVariable(Node node, mlir::Value value, mlir::Location loc) {
  auto str = getText(node);
  return declareVariable(str, value, loc);
}

mlir::FailureOr<mlir::Value> MLIRGen3::declareVariable(llvm::StringRef name, mlir::Value value, mlir::Location loc) {
  auto savedName = stringArena.save(name);
  DBGS("declaring '" << savedName << "' of type " << value.getType() << "\n");
  if (variables.count(savedName)) {
    return mlir::emitError(loc)
        << "Variable '" << name << "' already declared";
  }
  variables.insert(savedName, value);
  return value;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen3::gen() {
  TRACE();
  module = builder.create<mlir::ModuleOp>(loc(root), "ocamlc2");
  builder.setInsertionPointToStart(module->getBody());
  auto res = gen(root);
  if (mlir::failed(res)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    return mlir::failure();
  }

  if (mlir::failed(module->verify())) {
    DBGS("Failed to verify MLIR for compilation unit\n");
    llvm::errs() << *module << "\n";
    return mlir::failure();
  }

  return std::move(module);
}

#include "ocamlc2/Parse/MLIRGen3.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TSUtil.h"
#include <llvm/ADT/STLExtras.h>
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
#include "ocamlc2/Support/Utils.h"
#define DEBUG_TYPE "mlirgen"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;
using InsertionGuard = mlir::ocaml::OcamlOpBuilder::InsertionGuard;

using mlir::failed;
using mlir::failure;
using mlir::success;
using mlir::succeeded;
using mlir::LogicalResult;

static StringArena stringArena;

mlir::FailureOr<mlir::Value> MLIRGen3::genLetBinding(const ocamlc2::ts::Node node) {
  const auto children = getNamedChildren(node);
  const bool isRecursive = isLetBindingRecursive(node.getCursor());
  const auto lhs = children[0];
  const auto lhsType = unifierType(lhs);
  const auto rhs = children[children.size() - 1];
  if (lhsType == unifier.getUnitType()) {
    return gen(rhs);
  }
  return nyi(node) << " non-unit lhs";
  if (isRecursive) {
    return nyi(node);
  }
  return nyi(node);
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

mlir::FailureOr<mlir::Type> MLIRGen3::mlirFunctionType(const ocamlc2::ts::Node node) {
  const auto *type = unifierType(node);
  return mlirFunctionType(type, loc(node));
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirFunctionType(const ocamlc2::TypeExpr *type, mlir::Location loc) {
  if (type == nullptr) {
    return error(loc) << "Type for node was not inferred at unification-time";
  }
  if (const auto *to = llvm::dyn_cast<ocamlc2::TypeOperator>(type)) {
    const auto typeOperatorArgs = to->getArgs();
    if (to->getName() != TypeOperator::getFunctionOperatorName()) {
      return error(loc) << "Could not get MLIR function type from unified type: " << *type;
    }
    auto args = llvm::drop_end(typeOperatorArgs);
    auto argTypes = llvm::to_vector(llvm::map_range(args, [this, loc](auto *arg) {
      return *mlirType(arg, loc);
    }));
    auto returnType = *mlirType(typeOperatorArgs.back(), loc);
    return mlir::FunctionType::get(builder.getContext(), argTypes, {returnType});
  }
  return error(loc) << "Could get MLIR type from unified function type: "
                     << *type;
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(const ocamlc2::TypeExpr *type,
                                               mlir::Location loc) {
  if (const auto *to = llvm::dyn_cast<ocamlc2::TypeOperator>(type)) {
    const auto args = to->getArgs();
    if (args.empty()) {
      auto mlirType = mlirTypeFromBasicTypeOperator(to->getName());
      if (failed(mlirType)) {
        return error(loc) << "Unknown basic type operator: " << *type;
      }
      return mlirType;
    } else if (to->getName() == TypeOperator::getFunctionOperatorName()) {
      return mlirFunctionType(type, loc);
    }
    return error(loc) << "Unknown type operator: " << *type;
  }
  return error(loc) << "Unknown type: " << *type;
}

mlir::FailureOr<mlir::Type> MLIRGen3::mlirType(const ocamlc2::ts::Node node) {
  const auto *type = unifierType(node);
  if (type == nullptr) {
    return error(node) << "Type for node was not inferred at unification-time";
  }
  return mlirType(type, loc(node));
}


mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(const ocamlc2::ts::Node node) {
  auto str = getText(node);
  return getVariable(str, loc(node));
}

mlir::FailureOr<mlir::Value> MLIRGen3::getVariable(llvm::StringRef name, mlir::Location loc) {
  return variables.lookup(name);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genForExpression(const ocamlc2::ts::Node node) {
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
  auto res = declareVariable(inductionVariableNode, forOp.getInductionVar(),
                             loc(inductionVariableNode));
  if (failed(res)) {
    return res;
  }
  auto bodyValue = gen(bodyNode);
  if (failed(bodyValue)) {
    return error(node) << "Failed to generate body for `for` expression";
  }
  // for expressions are of unit type, don't need to propagate anything here
  return mlir::Value();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genCompilationUnit(const ocamlc2::ts::Node node) {
  VariableScope scope(variables);
  InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module->getBody());
  auto mainFunc = builder.create<mlir::func::FuncOp>(
      loc(node), "main", builder.getFunctionType({}, builder.getI32Type()));
  auto &body = mainFunc.getFunctionBody();
  builder.setInsertionPointToEnd(&body.emplaceBlock());
  for (auto child : getNamedChildren(node)) {
    auto res = gen(child);
    if (mlir::failed(res)) {
      return res;
    }
  }
  return mlir::Value();
}

mlir::FailureOr<mlir::Value> MLIRGen3::genNumber(const ocamlc2::ts::Node node) {
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

mlir::FailureOr<mlir::func::FuncOp> MLIRGen3::genCallee(const ocamlc2::ts::Node node) {
  if (node.getType() == "value_path") {
    auto str = getText(node);
    auto callee = module->lookupSymbol<mlir::func::FuncOp>(str);
    if (callee == nullptr) {
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
      return callee;
    }
    return callee;
  } else {
    return error(node) << "Unknown callee: " << node.getType();
  }
}

mlir::FailureOr<mlir::Value> MLIRGen3::genApplicationExpression(const ocamlc2::ts::Node node) {
  const auto children = getNamedChildren(node);
  const auto callee = children[0];
  auto calleeFunc = genCallee(callee);
  if (mlir::failed(calleeFunc)) {
    return failure();
  }
  auto args = llvm::drop_begin(children);
  llvm::SmallVector<mlir::Value> argValues = llvm::to_vector(llvm::map_range(
      args,
      [this](const ocamlc2::ts::Node &arg) { return *gen(arg); }));
  return builder.createCall(loc(node), *calleeFunc, argValues);
}

mlir::FailureOr<mlir::Value> MLIRGen3::gen(const ocamlc2::ts::Node node) {
  TRACE();
  auto type = node.getType();
  static constexpr std::string_view passthroughTypes[] = {
    "parenthesized_expression",
    "then_clause",
    "else_clause",
    "do_clause",
    "value_definition",
    "expression_item",
    "parenthesized_type",
  };
  if (llvm::is_contained(passthroughTypes, type)) {
    return gen(node.getNamedChild(0));
  } else if (type == "compilation_unit") {
    return genCompilationUnit(node);
  } else if (type == "let_binding") {
    return genLetBinding(node);
  } else if (type == "comment") {
    return mlir::Value();
  } else if (type == "number") {
    return genNumber(node);
  } else if (type == "for_expression") {
    return genForExpression(node);
  } else if (type == "application_expression") {
    return genApplicationExpression(node);
  } else if (type == "value_path") {
    return getVariable(node);
  }
  return error(node) << "Unknown node type: " << type;
}

llvm::StringRef MLIRGen3::getText(const ocamlc2::ts::Node node) {
  auto str = ::getText(node, unifier.source);
  return stringArena.save(str);
}

mlir::FailureOr<mlir::Value> MLIRGen3::declareVariable(ocamlc2::ts::Node node, mlir::Value value, mlir::Location loc) {
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
    return mlir::failure();
  }

  return std::move(module);
}

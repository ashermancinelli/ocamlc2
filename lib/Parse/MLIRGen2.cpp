#include "ocamlc2/Parse/MLIRGen2.h"
#include "ocamlc2/Parse/AST.h"
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

mlir::FailureOr<mlir::Value> MLIRGen2::gen(NumberExprAST const& node) {
  TRACE();
  StringRef textRef = node.getValue();
  int result;
  textRef.getAsInteger(10, result);
  auto op = builder.create<mlir::arith::ConstantIntOp>(loc(&node), result, 64);
  auto box = builder.createEmbox(loc(&node), op.getResult())->getResult(0);
  return box;
}

static mlir::FailureOr<std::string> getApplicatorName(ASTNode const& node) {
  if (auto *path = llvm::dyn_cast<ValuePathAST>(&node)) {
    return path->getPath().back();
  }
  if (auto *path = llvm::dyn_cast<ConstructorPathAST>(&node)) {
    return path->getPath().back();
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> MLIRGen2::genRuntime(llvm::StringRef name, ApplicationExprAST const& node) {
  TRACE();
  SmallVector<mlir::Value> convertedArgs;
  if (name == "print_int") {
    auto arg = gen(*node.getArgument(0));
    if (mlir::failed(arg)) {
      return mlir::emitError(loc(&node))
          << "Failed to generate argument for " << name;
    }
    auto type = builder.emboxType(builder.getI64Type());
    convertedArgs.push_back(
        builder.createConvert(loc(&node), *arg, type)->getResult(0));
    auto call = builder.createCallIntrinsic(loc(&node), "print_int", convertedArgs);
    return call->getResult(0);
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ApplicationExprAST const& node) {
  TRACE();
  auto maybeName = getApplicatorName(*node.getFunction());
  if (mlir::failed(maybeName)) {
    return mlir::emitError(loc(&node))
        << "Unknown AST node type: " << ASTNode::getName(node);
  }
  auto name = *maybeName;

  if (auto runtimeCall = genRuntime(name, node); succeeded(runtimeCall)) {
    return runtimeCall;
  }

  auto function = module->lookupSymbol<mlir::func::FuncOp>(name);
  if (!function) {
    return mlir::emitError(loc(&node))
        << "Applicator " << name << " not found";
  }
  auto ftype = function.getFunctionType();
  llvm::SmallVector<mlir::Value> args;
  for (size_t i = 0; i < node.getNumArguments(); ++i) {
    auto arg = gen(*node.getArgument(i));
    if (mlir::failed(arg)) {
      return mlir::emitError(loc(&node))
          << "Failed to generate argument " << i << " for applicator " << name;
    }
    args.push_back(*arg);
  }
  return builder.createCall(loc(&node), ftype, args)->getResult(0);
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ASTNode const& node) {
  if (auto *exprItem = llvm::dyn_cast<ExpressionItemAST>(&node)) {
    return gen(*exprItem);
  }
  else if (auto *valueDef = llvm::dyn_cast<ValueDefinitionAST>(&node)) {
    return gen(*valueDef);
  }
  else if (auto *application = llvm::dyn_cast<ApplicationExprAST>(&node)) {
    return gen(*application);
  }
  else if (auto *number = llvm::dyn_cast<NumberExprAST>(&node)) {
    return gen(*number);
  }
  DBGS("Unknown AST node type: " << ASTNode::getName(node) << "\n");
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ExpressionItemAST const& node) {
  TRACE();
  return gen(*node.getExpression());
}

mlir::FailureOr<mlir::Value> MLIRGen2::gen(CompilationUnitAST const& node) {
  TRACE();
  mlir::Location l = loc(&node);
  mlir::FunctionType mainFuncType =
      builder.getFunctionType({}, {builder.getI32Type()});

  auto mainFunc = builder.create<mlir::func::FuncOp>(
      l, "main", mainFuncType);
  auto *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  mlir::FailureOr<mlir::Value> res;
  for (auto &decl : node.getItems()) {
    TRACE();
    res = gen(*decl);
    if (mlir::failed(res)) {
      return mlir::failure();
    }
  }
  mlir::Value returnValue = builder.create<mlir::arith::ConstantIntOp>(l, 0, 32);
  builder.create<mlir::func::ReturnOp>(l, returnValue);
  return returnValue;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen2::gen() {
  TRACE();
  module = builder.create<mlir::ModuleOp>(loc(compilationUnit.get()), "ocamlc2");
  auto *ast = compilationUnit.get();
  auto compilationUnit = llvm::dyn_cast<CompilationUnitAST>(ast);
  if (!compilationUnit) {
    return mlir::emitError(loc(ast))
        << "Compilation unit not found";
  }
  builder.setInsertionPointToStart(module->getBody());
  auto res = gen(*compilationUnit);
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

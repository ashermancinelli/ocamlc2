#include "ocamlc2/Parse/MLIRGen3.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Parse/AST.h"
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

mlir::FailureOr<mlir::Value> MLIRGen3::genLetBinding(const ocamlc2::ts::Node node) {
  return nyi(node);
}

mlir::FailureOr<mlir::Value> MLIRGen3::genValueDefinition(const ocamlc2::ts::Node node) {
  return nyi(node);
}

mlir::FailureOr<mlir::Value> MLIRGen3::gen(const ocamlc2::ts::Node node) {
  TRACE();
  auto type = node.getType();
  static constexpr std::string_view passthroughTypes[] = {
    "parenthesized_expression",
    "then_clause",
    "else_clause",
    "value_definition",
    "expression_item",
    "parenthesized_type",
  };
  if (llvm::is_contained(passthroughTypes, type)) {
    return gen(node.getNamedChild(0));
  } else if (type == "compilation_unit") {
    for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
      auto child = node.getNamedChild(i);
      auto res = gen(child);
      if (mlir::failed(res)) {
        return res;
      }
    }
    return mlir::Value();
  } else if (type == "let_binding") {
    return genLetBinding(node);
  } else if (type == "value_definition") {
    return genValueDefinition(node);
  } else if (type == "comment") {
    return mlir::Value();
  }
  return error(node) << "Unknown node type: " << type;
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

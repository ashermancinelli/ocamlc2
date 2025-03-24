#pragma once
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Parse/TSAdaptor.h"
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

using Scope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using Node = std::pair<StringRef, TSNode>;
using NodeList = std::vector<Node>;
using NodeIter = NodeList::iterator;

class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder);
  FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen(TSTreeAdaptor &&adaptor);
  void genCompilationUnit(TSNode node);
  FailureOr<mlir::Value> gen(Node node);
  FailureOr<mlir::Value> gen(NodeIter &it);
  FailureOr<mlir::Value> gen(NodeList & nodes);
  FailureOr<mlir::Value> genLetBinding(TSNode node);
  FailureOr<mlir::Value> genAssign(Node lhs, mlir::Value rhs);
  mlir::Location loc(TSNode node);
private:
  TSTreeAdaptor *adaptor;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  [[maybe_unused]] mlir::MLIRContext &context;
  mlir::OpBuilder &builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

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

class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder);
  FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen(TSTreeAdaptor adaptor);
private:
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  [[maybe_unused]] mlir::MLIRContext &context;
  mlir::OpBuilder &builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

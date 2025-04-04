
#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include <llvm/ADT/ScopedHashTable.h>

struct MLIRGen2;
struct TypeConstructor {
  using FunctionType = std::function<
    mlir::Type( /* Create a type */
      MLIRGen2 &, 
      llvm::ArrayRef<mlir::Type> /* With these type parameters */
    )
  >;
  FunctionType constructor;
};
using TypeConstructorScope = llvm::ScopedHashTableScope<llvm::StringRef, TypeConstructor>;

struct MLIRGen2 {
  MLIRGen2(mlir::MLIRContext &context, std::unique_ptr<ocamlc2::ASTNode> compilationUnit) 
    : context(context), compilationUnit(std::move(compilationUnit)) {}
  llvm::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen();
private:
  mlir::MLIRContext &context;
  std::unique_ptr<ocamlc2::ASTNode> compilationUnit;
};


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
  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen();
  inline mlir::Location loc(const ocamlc2::ASTNode *node) const {
    return node->getMLIRLocation(context);
  }
private:
  mlir::MLIRContext &context;
  std::unique_ptr<ocamlc2::ASTNode> compilationUnit;
};

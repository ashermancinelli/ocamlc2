#include "ocamlc2/Parse/MLIRGen2.h"
#include "ocamlc2/Parse/AST.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#define DEBUG_TYPE "mlirgen"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;

mlir::FailureOr<mlir::Value> MLIRGen2::gen(ASTNode const& node) {
  return llvm::TypeSwitch<ASTNode const*, mlir::FailureOr<mlir::Value>>(&node)
    .Case<ExpressionItemAST>([&](auto *node) mutable {
      return gen(*node->getExpression());
    })
    .Default([&](auto *node) {
      DBGS("Unknown AST node type: " << node->getKind() << "\n");
      return mlir::failure();
    });
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen2::gen() {
  mlir::MLIRContext &context = this->context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module = builder.create<mlir::ModuleOp>(loc(compilationUnit.get()), "ocamlc2");
  auto *ast = compilationUnit.get();
  auto res = llvm::TypeSwitch<ASTNode*, mlir::LogicalResult>(ast)
    .Case<CompilationUnitAST>([&](auto *node) mutable -> mlir::LogicalResult {
      for (auto &decl : node->getItems()) {
        DBGS("gen declaration\n");
        if (mlir::failed(gen(*decl))) {
          return mlir::failure();
        }
      }
      return mlir::success();
    })
    .Default([&](auto *node) {
      DBGS("Unknown AST node type: " << node->getKind() << "\n");
      return mlir::failure();
    });

  if (mlir::failed(res)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    return mlir::failure();
  }

  if (mlir::failed(module->verify())) {
    DBGS("Failed to verify MLIR for compilation unit\n");
    return mlir::failure();
  }

  return module;
}

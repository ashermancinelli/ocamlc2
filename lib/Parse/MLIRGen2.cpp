#include "ocamlc2/Parse/MLIRGen2.h"
#include <mlir/IR/Builders.h>

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen2::gen() {
  mlir::MLIRContext &context = this->context;
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module = builder.create<mlir::ModuleOp>(loc(compilationUnit.get()), "ocamlc2");
  return module;
}

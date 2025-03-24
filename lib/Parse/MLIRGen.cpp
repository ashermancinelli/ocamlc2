#include "ocamlc2/Parse/MLIRGen.h"
#include <mlir/IR/Location.h>
#include <mlir/IR/Attributes.h>

MLIRGen::MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder)
    : context(context), builder(builder) {
}

FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> MLIRGen::gen(TSTreeAdaptor adaptor) {
  auto filenameAttr = builder.getStringAttr(adaptor.getFilename());
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(mlir::FileLineColLoc::get(filenameAttr, 0, 0));
  return std::move(module);
}

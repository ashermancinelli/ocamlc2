#pragma once

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/ValueRange.h>

namespace mlir::ocaml {

class OcamlOpBuilder : public mlir::OpBuilder {
  using OpBuilder::OpBuilder;

public:
  mlir::Operation *createConvert(mlir::Location loc, mlir::Value input, mlir::Type resultType) {
    return create<mlir::ocaml::ConvertOp>(loc, resultType, input);
  }

  mlir::Operation *createEmbox(mlir::Location loc, mlir::Value input) {
    return createConvert(loc, input, emboxType(input.getType()));
  }

  mlir::Operation *createCall(mlir::Location loc, mlir::FunctionType ftype, mlir::ValueRange args) {
    SmallVector<mlir::Value> convertedArgs;
    for (auto arg : llvm::enumerate(args)) {
      convertedArgs.push_back(
          createConvert(loc, arg.value(), ftype.getInput(arg.index()))
              ->getResult(0));
    }
    return create<mlir::func::CallOp>(loc, ftype, convertedArgs);
  }

  mlir::Operation *createCallIntrinsic(mlir::Location loc, StringRef callee, mlir::ValueRange args) {
    return create<mlir::ocaml::IntrinsicOp>(loc, getOBoxType(), getStringAttr(callee), args);
  }

  mlir::Operation *createCallIntrinsic(mlir::Location loc, StringRef callee, mlir::ValueRange args, mlir::Type resultType) {
    return create<mlir::ocaml::IntrinsicOp>(loc, resultType, getStringAttr(callee), args);
  }

  inline mlir::Type emboxType(mlir::Type elementType) {
    return mlir::ocaml::BoxType::get(elementType);
  }

  inline mlir::Type getOBoxType() {
    return mlir::ocaml::OpaqueBoxType::get(getContext());
  }

  inline mlir::Type getUnitType() {
    return mlir::ocaml::UnitType::get(getContext());
  }
};

}

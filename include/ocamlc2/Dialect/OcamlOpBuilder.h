#pragma once

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include <mlir/IR/ValueRange.h>

namespace mlir::ocaml {

class OcamlOpBuilder : public mlir::OpBuilder {
  using OpBuilder::OpBuilder;

public:
  mlir::Value createConvert(mlir::Location loc, mlir::Value input, mlir::Type resultType) {
    return create<mlir::ocaml::ConvertOp>(loc, resultType, input).getResult();
  }

  mlir::Value createEmbox(mlir::Location loc, mlir::Value input) {
    return createConvert(loc, input, emboxType(input.getType()));
  }

  mlir::Value createCallIntrinsic(mlir::Location loc, StringRef callee, mlir::ValueRange args) {
    return create<mlir::ocaml::IntrinsicOp>(loc, getOBoxType(), getStringAttr(callee), args).getResult();
  }

  mlir::Value createCallIntrinsic(mlir::Location loc, StringRef callee, mlir::ValueRange args, mlir::Type resultType) {
    return create<mlir::ocaml::IntrinsicOp>(loc, resultType, getStringAttr(callee), args).getResult();
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

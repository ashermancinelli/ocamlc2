#pragma once

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "ocamlc2/Dialect/OcamlDialect.h"

namespace mlir::ocaml {

class OcamlOpBuilder : public mlir::OpBuilder {
  using OpBuilder::OpBuilder;

public:
  mlir::Value createConvert(mlir::Location loc, mlir::Value input, mlir::Type resultType) {
    return create<mlir::ocaml::ConvertOp>(loc, resultType, input).getResult();
  }

  mlir::Value createEmbox(mlir::Location loc, mlir::Value input) {
    return createConvert(loc, input, mlir::ocaml::BoxType::get(input.getType()));
  }
};

}

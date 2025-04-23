#pragma once

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/ValueRange.h>

namespace mlir::ocaml {

class OcamlOpBuilder : public mlir::OpBuilder {
  using OpBuilder::OpBuilder;

public:
  mlir::Value createConvert(mlir::Location loc, mlir::Value input, mlir::Type resultType) {
    return create<mlir::ocaml::ConvertOp>(loc, resultType, input);
  }

  mlir::Value createEmbox(mlir::Location loc, mlir::Value input) {
    return createConvert(loc, input, emboxType(input.getType()));
  }

  mlir::Value createUnbox(mlir::Location loc, mlir::Value input) {
    auto type = input.getType();
    if (auto boxType = mlir::dyn_cast<mlir::ocaml::BoxType>(type)) {
      return createConvert(loc, input, boxType.getElementType());
    }
    return input;
  }

  mlir::Value createConstant(mlir::Location loc, mlir::Type type, int64_t value) {
    return create<mlir::arith::ConstantOp>(loc, type, mlir::IntegerAttr::get(type, value));
  }

  mlir::Value createCall(mlir::Location loc, func::FuncOp function, mlir::ValueRange args) {
    SmallVector<mlir::Value> convertedArgs;
    auto ftype = function.getFunctionType();
    for (auto arg : llvm::enumerate(args)) {
      convertedArgs.push_back(
          createConvert(loc, arg.value(), ftype.getInput(arg.index())));
    }
    return create<mlir::func::CallOp>(loc, function, convertedArgs).getResult(0);
  }

  mlir::Value createCallIntrinsic(mlir::Location loc, StringRef callee, mlir::ValueRange args) {
    return create<mlir::ocaml::IntrinsicOp>(loc, getUnitType(), getStringAttr(callee), args);
  }

  mlir::Value createCallIntrinsic(mlir::Location loc, StringRef callee, mlir::ValueRange args, mlir::Type resultType) {
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

mlir::FailureOr<mlir::Type> resolveTypes(mlir::Type lhs, mlir::Type rhs, mlir::Location loc);
mlir::FailureOr<std::string> getPODTypeRuntimeName(mlir::Type type);
mlir::FailureOr<std::string> binaryOpToRuntimeName(std::string op, mlir::Location loc);

}

#pragma once

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/ValueRange.h>

namespace mlir::ocaml {

class OcamlOpBuilder : public mlir::OpBuilder {
  using OpBuilder::OpBuilder;

public:
  mlir::Value createPatternVariable(mlir::Location loc, mlir::Type type) {
    return create<mlir::ocaml::PatternVariableOp>(loc, type);
  }

  mlir::Value createPatternMatch(mlir::Location loc, mlir::Value scrutinee, mlir::Value pattern) {
    pattern = createConvert(loc, scrutinee.getType(), pattern);
    return create<mlir::ocaml::PatternMatchOp>(loc, getI1Type(), scrutinee, pattern);
  }

  mlir::Value createString(mlir::Location loc, llvm::StringRef str) {
    return create<mlir::ocaml::EmboxStringOp>(
        loc, StringType::get(getContext()), StringAttr::get(getContext(), str));
  }

  mlir::Value createConvert(mlir::Location loc, mlir::Value input, mlir::Type resultType) {
    if (resultType == input.getType()) {
      return input;
    }
    return create<mlir::ocaml::ConvertOp>(loc, resultType, input);
  }
  mlir::Value createConvert(mlir::Location loc, mlir::Type resultType, mlir::Value input) {
    if (resultType == input.getType()) {
      return input;
    }
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

  SmallVector<mlir::Value> prepareArguments(mlir::Location loc, mlir::FunctionType functionType, mlir::ValueRange args) {
    return llvm::map_to_vector(llvm::enumerate(args), [this, loc, functionType](auto arg) {
      return createConvert(loc, arg.value(), functionType.getInput(arg.index()));
    });
  }

  mlir::Value createCall(mlir::Location loc, func::FuncOp function, mlir::ValueRange args) {
    return create<mlir::func::CallOp>(
               loc, function,
               prepareArguments(loc, function.getFunctionType(), args))
        .getResult(0);
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

  inline mlir::Type getArrayType(mlir::Type elementType) {
    return mlir::ocaml::ArrayType::get(elementType);
  }

  inline mlir::Type getOBoxType() {
    return mlir::ocaml::OpaqueBoxType::get(getContext());
  }

  inline mlir::Type getUnitType() {
    return mlir::ocaml::UnitType::get(getContext());
  }

  inline mlir::Type getStringType() {
    return mlir::ocaml::StringType::get(getContext());
  }

  mlir::Type getVariantType(llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> constructors, llvm::ArrayRef<mlir::Type> types);
  mlir::Type getTupleType(llvm::ArrayRef<mlir::Type> types);

  mlir::Value createUnit(mlir::Location loc) {
    return create<mlir::ocaml::UnitOp>(loc, getUnitType());
  }

  llvm::SmallVector<mlir::StringAttr> createStringAttrVector(llvm::ArrayRef<llvm::StringRef> strings);
  mlir::NamedAttribute createVariantCtorAttr();
};

mlir::FailureOr<mlir::Type> resolveTypes(mlir::Type lhs, mlir::Type rhs, mlir::Location loc);
mlir::FailureOr<std::string> getPODTypeRuntimeName(mlir::Type type);
mlir::FailureOr<std::string> binaryOpToRuntimeName(std::string op, mlir::Location loc);

}

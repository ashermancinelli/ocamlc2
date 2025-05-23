#pragma once
#include "OcamlDialect.h"

namespace mlir::ocaml {

bool areTypesCoercible(mlir::Type from, mlir::Type into);

template<typename ISA>
struct ConvertibleValue : public mlir::Value {
  inline operator bool() const {
    return mlir::isa<ISA>(getType());
  }
  inline operator LogicalResult() const {
    return success(*this);
  }
};
struct Closure : public mlir::Value {
  inline operator bool() const {
    if (auto boxType = mlir::dyn_cast<BoxType>(getType())) {
      return mlir::isa<FunctionType>(boxType.getElementType());
    }
    return false;
  }
  inline operator LogicalResult() const {
    return success(*this);
  }
  inline FunctionType getFunctionType() const {
    assert(*this);
    return mlir::cast<FunctionType>(mlir::cast<BoxType>(getType()).getElementType());
  }
};

struct ClosureEnvValue : public mlir::Value {
  ClosureEnvValue(mlir::Value value) : mlir::Value(value) {}
  inline operator bool() const {
    return mlir::isa<EnvType>(getType());
  }
  inline operator LogicalResult() const {
    return success(*this);
  }
  StringAttr getFor() const {
    assert(*this);
    return mlir::cast<mlir::StringAttr>(getDefiningOp()->getAttr("for"));
  }
};
}

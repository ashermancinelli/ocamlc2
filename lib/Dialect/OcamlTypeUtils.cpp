#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlTypeUtils.h"

namespace mlir::ocaml {

bool areTypesCoercible(mlir::Type from, mlir::Type into) {
  if (from == into) {
    return true;
  }
  if (mlir::isa<mlir::ocaml::OpaqueBoxType>(into)) {
    return true;
  }
  if (mlir::isa<mlir::ocaml::ArrayType>(from) && mlir::isa<mlir::ocaml::ArrayType>(into)) {
    auto fromArray = mlir::cast<mlir::ocaml::ArrayType>(from);
    auto intoArray = mlir::cast<mlir::ocaml::ArrayType>(into);
    return areTypesCoercible(fromArray.getElementType(), intoArray.getElementType());
  }
  return false;
}

bool isa_box_type(mlir::Type type) {
  return isa<BoxType, OpaqueBoxType, StringType, UnitType, ClosureType>(type);
}

}

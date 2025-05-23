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
  return false;
}

}

#pragma once
#include "OcamlDialect.h"

namespace mlir::ocaml {

bool areTypesCoercible(mlir::Type from, mlir::Type into);
bool isa_box_type(mlir::Type type);

}

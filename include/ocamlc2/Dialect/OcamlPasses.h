#pragma once
#include "mlir/Pass/Pass.h"
#include "OcamlDialect.h"

namespace mlir {
namespace ocaml {

#define GEN_PASS_DECL
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

} // namespace ocaml
} // namespace mlir

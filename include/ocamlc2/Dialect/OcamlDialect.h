#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "ocamlc2/Dialect/OcamlDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ocamlc2/Dialect/OcamlTypes.h.inc"

#define GET_OP_CLASSES
#include "ocamlc2/Dialect/OcamlOps.h.inc"

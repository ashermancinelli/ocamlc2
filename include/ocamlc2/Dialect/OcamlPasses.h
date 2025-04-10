#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ocaml {

#define GEN_PASS_DECL
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "ocamlc2/Dialect/OcamlPasses.h.inc"

} // namespace ocaml
} // namespace mlir

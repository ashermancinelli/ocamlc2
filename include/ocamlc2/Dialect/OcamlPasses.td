#ifndef OCAML_DIALECT_OCAML_PASSES_H
#define OCAML_DIALECT_OCAML_PASSES_H

include "mlir/Pass/PassBase.td"
include "mlir/Dialect/LLVMIR/LLVMDialect.td"

def BufferizeBoxes : Pass<"bufferize-boxes", "mlir::ModuleOp"> {
  let summary = "Bufferize OCaml boxes";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

def OcamlLowerToLLVM : Pass<"convert-ocaml-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert OCaml to LLVM";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

#endif

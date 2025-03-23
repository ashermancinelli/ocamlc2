
add_mlir_dialect(OCamlOps ocaml)

set(LLVM_TARGET_DEFINITIONS OCamlPasses.td)
mlir_tablegen(OCamlPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIROcamlPassesIncGen)

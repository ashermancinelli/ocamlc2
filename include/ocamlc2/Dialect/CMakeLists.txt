
set(LLVM_TARGET_DEFINITIONS OcamlDialect.td)
mlir_tablegen(OcamlOps.h.inc -gen-op-decls)
mlir_tablegen(OcamlOps.cpp.inc -gen-op-defs)
mlir_tablegen(OcamlTypes.h.inc -gen-typedef-decls)
mlir_tablegen(OcamlTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(OcamlDialect.h.inc -gen-dialect-decls)
mlir_tablegen(OcamlDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(OcamlIncGen)

# set(LLVM_TARGET_DEFINITIONS OcamlPasses.td)
# mlir_tablegen(OcamlPasses.h.inc --gen-pass-decls)
# add_public_tablegen_target(OcamlPassesIncGen)

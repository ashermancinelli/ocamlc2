#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "ocamlc2/Dialect/OcamlDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::ocaml::setupRegistry(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}

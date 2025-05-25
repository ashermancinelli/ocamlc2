#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::ocaml::setupRegistry(registry);
  return failed(MlirOptMain(argc, argv, "OCaml optimizer driver\n", registry));
}

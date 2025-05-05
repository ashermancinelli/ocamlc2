#include "mlir/IR/MLIRContext.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"

int main() {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  mlir::ocaml::setupRegistry(registry);
  context.appendDialectRegistry(registry);
  mlir::ocaml::setupContext(context);
  mlir::ocaml::OcamlOpBuilder builder(&context);

  auto i64 = builder.getI64Type();
  auto unit = builder.getUnitType();
  auto obox = builder.getOBoxType();
  auto box = builder.emboxType(i64);
  auto tuple = builder.getTupleType({i64, i64});

  llvm::outs() << unit << "\n" << obox << "\n" << box << "\n" << tuple << "\n";
  // CHECK: !ocaml.unit
  // CHECK: !ocaml.obox
  // CHECK: !ocaml.box<i64>
  // CHECK: tuple<i64, i64>

  auto variant = builder.getVariantType("foo", {"None", "Some", "More"},
                                        {unit, i64, tuple});
  llvm::outs() << variant << "\n";
  // CHECK: !ocaml.variant<"foo" is "None" | "Some" of i64 | "More" of tuple<i64, i64>>

  return 0;
}



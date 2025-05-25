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

  auto variant = builder.getVariantType("foo", {"None", "Some", "More"},
                                        {unit, i64, tuple});
  llvm::outs() << variant << "\n";

  auto array = builder.getArrayType(i64);
  llvm::outs() << array << "\n";

  auto functionType = builder.getFunctionType(i64, i64);
  auto closure = mlir::ocaml::ClosureType::get(functionType);
  llvm::outs() << closure << "\n";

  {
    auto list = mlir::ocaml::ListType::get(i64);
    llvm::outs() << list << "\n";
  }
  {
    auto list = mlir::ocaml::ListType::get(closure);
    llvm::outs() << list << "\n";
  }

  auto env = mlir::ocaml::EnvType::get(&context);
  llvm::outs() << env << "\n";

  return 0;
}

// CHECK: !ocaml.unit
// CHECK: !ocaml.obox
// CHECK: !ocaml.box<i64>
// CHECK: !ocaml.tuple<i64, i64>
// CHECK: !ocaml.variant<"foo" is "None" | "Some" of i64 | "More" of !ocaml.tuple<i64, i64>>
// CHECK: !ocaml.array<i64>
// CHECK: !ocaml.closure<(i64) -> i64>
// CHECK: !ocaml.list<i64>
// CHECK: !ocaml.list<!ocaml.closure<(i64) -> i64>>
// CHECK: !ocaml.env


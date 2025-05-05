#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include <llvm/ADT/TypeSwitch.h>
#include <string>
using namespace std::string_literals;

namespace mlir::ocaml {

FailureOr<mlir::Type> resolveTypes(mlir::Type lhs, mlir::Type rhs, mlir::Location loc) {
  if (lhs == rhs) {
    return lhs;
  }
  return mlir::emitError(loc) << "I don't know how to resolve these types! " << lhs << " and " << rhs;
}

FailureOr<std::string> getPODTypeRuntimeName(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, FailureOr<std::string>>(type)
      .Case<mlir::Float64Type>([](auto) -> std::string { return "f64"; })
      .Case<mlir::IntegerType>([](auto) -> std::string { return "i64"; })
      .Default([](auto) -> LogicalResult { return failure(); });
}
mlir::FailureOr<std::string> binaryOpToRuntimeName(std::string op, mlir::Location loc) {
  if (op == "+") {
    return "add"s;
  } else if (op == "-") {
    return "sub"s;
  } else if (op == "*") {
    return "mul"s;
  } else if (op == "/") {
    return "div"s;
  } else if (op == "%") {
    return "mod"s;
  }
  return mlir::emitError(loc) << "Unknown binary operator: " << op;
}

llvm::SmallVector<mlir::StringAttr> OcamlOpBuilder::createStringAttrVector(llvm::ArrayRef<llvm::StringRef> strings) {
  return llvm::map_to_vector(strings, [this](llvm::StringRef str) {
    return mlir::StringAttr::get(getContext(), str);
  });
}

mlir::Type OcamlOpBuilder::getVariantType(llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> constructors, llvm::ArrayRef<mlir::Type> types) {
  auto c = createStringAttrVector(constructors);
  auto n = getStringAttr(name);
  return mlir::ocaml::VariantType::get(getContext(), n, c, types);
}

}

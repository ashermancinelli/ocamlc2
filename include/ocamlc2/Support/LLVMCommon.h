#pragma once
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include <optional>
#include <vector>
#include <string_view>
#include "llvm/ADT/ArrayRef.h"

using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::FailureOr;
using llvm::succeeded;
using llvm::failed;
using llvm::success;
using llvm::failure;
using mlir::LogicalResult;
using llvm::StringRef;
using std::optional;
using std::nullopt;

inline auto must(auto &&result, std::string message="Unexpected failure") -> decltype(auto) {
  if (failed(result)) {
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return std::forward<decltype(result.value())>(result.value());
}
inline void must(LogicalResult result, std::string message="Unexpected failure") {
  if (failed(result)) {
    llvm::report_fatal_error(llvm::StringRef(message));
  }
}

std::string getUniqueName(StringRef prefix);

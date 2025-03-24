#pragma once
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/StringRef.h"
using llvm::FailureOr;
using llvm::succeeded;
using llvm::failed;
using llvm::success;
using llvm::failure;
using llvm::StringRef;

inline auto must(auto &&result, std::string message="Unexpected failure") -> decltype(auto) {
  if (failed(result)) {
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return std::forward<decltype(result.value())>(result.value());
}

std::string getUniqueName(StringRef prefix);

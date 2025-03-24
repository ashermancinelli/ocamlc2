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

template <typename T>
inline T must(FailureOr<T> result, llvm::StringRef message) {
  if (failed(result)) {
    llvm::report_fatal_error(message);
  }
  return result.get();
}
template <typename T>
inline T must(FailureOr<T> result) {
  if (failed(result)) {
    llvm::report_fatal_error("Failed to execute operation");
  }
  return result.value();
}

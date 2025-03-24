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
  return result.value();
}
template <typename T>
inline T must(FailureOr<T&&> result, llvm::StringRef message) {
  if (failed(result)) {
    llvm::report_fatal_error(message);
  }
  return std::move(result.value());
}
template <typename T>
inline T must(FailureOr<T> result) {
  return must(result, "");
}
template <typename T>
inline T must(FailureOr<T&&> result) {
  return must(std::move(result), "");
}

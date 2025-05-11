#pragma once

#include <functional>
#include <mlir/Support/LogicalResult.h>

namespace ocamlc2 {

template <typename T, typename F, typename U = typename std::invoke_result_t<F, T>>
U bind(mlir::FailureOr<T> v, F &&f) {
  if (mlir::failed(v)) {
    return mlir::failure();
  }
  return f(*v);
}

} // namespace ocamlc2

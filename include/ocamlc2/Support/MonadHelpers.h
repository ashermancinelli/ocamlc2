#pragma once

#include <functional>
#include <mlir/Support/LogicalResult.h>
#include <type_traits>

namespace ocamlc2 {

template <typename T, typename F, typename U = typename std::invoke_result_t<F, T>>
U bind(mlir::FailureOr<T> v, F &&f) {
  if (mlir::failed(v)) {
    return mlir::failure();
  }
  return f(*v);
}

template <typename T, typename F>
auto andThen(mlir::FailureOr<T> value, F &&f) 
    -> decltype(f(std::declval<T>())) {
  if (mlir::failed(value)) {
    return mlir::failure();
  }
  return f(*value);
}

// #if 0  // Disabled old tag-based helpers in favour of function-style wrappers
// inline constexpr struct AndThenTag { } and_then;
// inline constexpr struct OrElseTag { } or_else;
// #endif

// --- Begin new function-style monadic helpers --------------------------------

// Wrapper that stores the continuation for a successful value.
template <typename F>
struct AndThenWrapper {
  F func;
};

// Wrapper that stores the alternative computation for a failure.
template <typename F>
struct OrElseWrapper {
  F func;
};

// Factory helpers that create the wrappers.  These are the *functions* users
// will write:  `value | and_then(lambda)` or `value | or_else(lambda)`.

template <typename F>
auto and_then(F &&f) -> AndThenWrapper<std::decay_t<F>> {
  return {std::forward<F>(f)};
}

template <typename F>
auto or_else(F &&f) -> OrElseWrapper<std::decay_t<F>> {
  return {std::forward<F>(f)};
}

// Apply `and_then`: run the continuation only if the FailureOr succeeded.

template <typename T, typename F,
          typename R = decltype(std::declval<F &&>()(*std::declval<mlir::FailureOr<T>>() ))>
auto operator|(mlir::FailureOr<T> value, AndThenWrapper<F> wrapper) -> R {
  if (mlir::failed(value)) {
    return mlir::failure();
  }
  return wrapper.func(*value);
}

// Apply `or_else`: run the alternative only if the FailureOr failed.

template <typename T, typename F,
          typename R = typename std::invoke_result_t<F>>
auto operator|(mlir::FailureOr<T> value, OrElseWrapper<F> wrapper) -> R {
  if (mlir::succeeded(value)) {
    return value; // OK because R should be mlir::FailureOr<T>
  }
  return wrapper.func();
}

// --- End new function-style monadic helpers ----------------------------------

} // namespace ocamlc2

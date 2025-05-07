#pragma once

#include <functional>
namespace ocamlc2 {

template <typename T>
T* bind(T *v, std::function<T*(T*)> f) {
  if (!v) {
    return nullptr;
  }
  return f(v);
}

template <typename T, typename... Ts>
T *lift(Ts... args, std::function<T*(Ts...)> f) {
  return f(args...);
}

} // namespace ocamlc2

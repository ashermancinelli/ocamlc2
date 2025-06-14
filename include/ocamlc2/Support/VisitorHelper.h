#pragma once

namespace ocamlc2 {

template<typename ... Ts>
struct Overload : Ts ... {
  using Ts::operator() ...;
};
template<class... Ts> Overload(Ts...) -> Overload<Ts...>;

} // namespace ocamlc2

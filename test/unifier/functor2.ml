(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module M : OrderedType = struct
  type t = int
  let compare a b = if a < b then -1 else if a > b then 1 else 0
end

module N : OrderedType = struct
  include M
end

let f x:M.t = x;;
let g x:N.t = x;;

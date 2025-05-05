(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module M : OrderedType = struct
  type t = int
  let compare a b = 0;;
end

module N : OrderedType = struct
  include M
end

module O : sig type t val compare : t -> t -> int end = struct
  type t = int
  let compare a b = 0
end

let f x:M.t = x;;
let g x:N.t = x;;
let h x:O.t = x;;

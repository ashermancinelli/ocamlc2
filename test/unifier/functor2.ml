(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
*)
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module Make : functor (Ord : OrderedType) -> Set.S

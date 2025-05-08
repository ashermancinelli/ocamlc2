(*

Basics of the set module.

RUN: p3 -d -f %s

*)

module type ORDERED = sig
  type t

  val compare : t -> t -> int
end

module type S = sig
  type element_type
  type set_type

  val empty : set_type
end

module Make (Dep : ORDERED) = struct
  type element_type = Dep.t
  type set_type = element_type array

  let compare = Dep.compare
  let empty = [||]
end

module Int = struct
  type t = int

  let compare a b = a - b
end

module Float = struct
  type t = float

  let epsilon = 1.0e-6
  let compare a b = if a -. b < epsilon then 0 else if a < b then -1 else 1
end

module IntSet = Make (Int)
module FloatSet = Make (Float)

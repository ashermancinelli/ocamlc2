
(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
I'm not actually sure this is a good test. Extrapolated from stdlib.
*)

module String = struct
  type t = string
  let compare (x : t) (y : t) : int = 0
end

module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

(* The Map functor takes a module conforming to OrderedType *)
module Make (Ord: OrderedType) (Val: sig type t end) : sig
  type key = Ord.t
  type t = (key * Val.t) array
  
  val empty : t
  val add : key -> Val.t -> t -> t
  val find : key -> t -> Val.t
end

(* Define a module for our key type with a compare function *)
module StringKey = struct
  type t = string
  let compare = String.compare;;
end

(* Apply the functor to create a string-keyed map *)
module StringMap = Make(StringKey)(struct type t = int end)

(* Use the generated map module *)
let m = StringMap.empty
let m = StringMap.add "hello" 42 m
let value = StringMap.find "hello" m  (* value = 42 *)


(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
*)
(* First, define a module signature with the requirements *)
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

(* The Map functor takes a module conforming to OrderedType *)
module Make (Ord: OrderedType) : sig
  type key = Ord.t
  type 'a t
  
  val empty : 'a t
  val add : key -> 'a -> 'a t -> 'a t
  val find : key -> 'a t -> 'a
end

(* Define a module for our key type with a compare function *)
module StringKey = struct
  type t = string
  let compare = String.compare
end

(* Apply the functor to create a string-keyed map *)
module StringMap = Map.Make(StringKey)

(* Use the generated map module *)
let m = StringMap.empty
let m = StringMap.add "hello" 42 m
let value = StringMap.find "hello" m  (* value = 42 *)

(*
RUN: p3 --dtypes %s | FileCheck %s.ref
*)
type 'a array
val t1 : unit -> unit
val t2 : string -> unit
val t3 : 'a -> unit
val map : ('a -> 'b) -> 'a list -> 'b list
external get : 'a -> int -> 'a = "array_safe_get"
val ( ^ ) : string -> string -> string
type 'a arrayalias = 'a array
val t4 : 'a arrayalias -> 'a
val t5 : int arrayalias -> int
type iarray = int array
val t6 : iarray -> int
type 'a t = 'a option = None | Some of 'a
val t7 : 'a t -> 'a
val t8 : int t -> float
type t2 = int t
val t9 : t2 -> int option

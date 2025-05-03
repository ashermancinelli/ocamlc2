(*
RUN: p3 --dtypes %s | FileCheck %s.ref
*)
type intarrayalias = int array
val t1 : unit -> unit
val t2 : string -> unit
val t3 : 'a -> unit
val map : ('a -> 'b) -> 'a list -> 'b list
external get : 'a -> int -> 'a = "array_safe_get"
val ( ^ ) : string -> string -> string
type 'a arrayalias = 'a array
val t4 : 'a arrayalias -> 'a
val t5 : int arrayalias -> int
type intarrayalias2 = int array
val t6 : intarrayalias2 -> int
type 'a optionalias = 'a option = None | Some of 'a
val t7 : 'a optionalias -> 'a
val t8 : int optionalias -> float
type intoptionalias = int optionalias
val t9 : intoptionalias -> int option

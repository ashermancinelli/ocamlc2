(*
RUN: p3 --freestanding --dump-types %s | FileCheck %s
CHECK: val: t1 : (λ unit unit)
CHECK: val: t2 : (λ string unit)
CHECK: val: t3 : (λ '{{.+}} unit)
CHECK: val: map : (λ (λ '[[T1:.+]] '[[T2:.+]]) (list '[[T1]]) (list '[[T2]]))
CHECK: val: get : (λ '[[T:.+]] int '[[T]])
*)
val t1 : unit -> unit
val t2 : string -> unit
val t3 : 'a -> unit
val map : ('a -> 'b) -> 'a list -> 'b list
external get : 'a -> int -> 'a = "%array_safe_get"
val ( ^ ) : string -> string -> string
type iarray = int array


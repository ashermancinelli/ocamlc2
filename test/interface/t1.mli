(*
RUN: p3 --dump-types %s | FileCheck %s
CHECK: val: t1 : (λ unit unit)
CHECK: val: t2 : (λ string unit)
CHECK: val: t3 : (λ '{{.+}} unit)
CHECK: val: map : (λ (λ '[[T1:.+]] '[[T2:.+]]) (List '[[T1]]) (List '[[T2]]))
*)
val t1 : unit -> unit
val t2 : string -> unit
val t3 : 'a -> unit
val map : ('a -> 'b) -> 'a list -> 'b list

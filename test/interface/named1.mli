(*
RUN: p3 --freestanding --dump-types %s | FileCheck %s.ref
*)

type 'a t = 'a option = None | Some of 'a
val fold : ?none:'a -> some:('b -> 'a) -> 'b option -> 'a
val value : 'a option -> default:'a -> 'a
val to_result : none:'e -> 'a option -> ('a, 'e) result

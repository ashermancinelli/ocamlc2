(*
RUN: g3 %s | FileCheck %s.ref
*)
external foo : int -> int = "foo_bindc"
let bar = foo 10;;

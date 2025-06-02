(* Test global value definitions *)
(*
RUN: g3 %s | FileCheck %s.ref
 *)
let x = 42
let x = x + 1
let x = x * x

(* Test global value definitions *)
(*
RUN: g3 %s | FileCheck %s.ref
XFAIL: *
 *)
let x = 42
let y = x + 1
let z = "hello" 
let z = 5

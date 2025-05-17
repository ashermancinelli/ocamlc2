(*
 * RUN: g3 %s | FileCheck %s
 * XFAIL: *
 *)
let x = 5 in
let y = fun () -> x in
y ();;

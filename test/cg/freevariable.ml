(*
 * RUN: g3 %s | FileCheck %s.ref
 *)
let x = 5 in
let y = fun () -> x in
y ();;

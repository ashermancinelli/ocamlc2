(*
 * RUN: g3 %s | FileCheck %s.ref
 *)
let x = 5 in
let y = 7 in
let z () = x + y in
z ();;

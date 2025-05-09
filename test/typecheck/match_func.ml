(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)
let x = function
  | 0 -> 1
  | 1 -> 2
  | _ -> 3

type v = A | B of int
let y = function
  | A -> 1
  | B x -> x;;

let z = y @@ B 0;;
let a = y A;;

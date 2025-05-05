
(* Recursive function *)
let rec factorial n =
  if n <= 1 then 1
  else n * factorial (n - 1);;

let x = factorial 10;;
let f = factorial;;

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

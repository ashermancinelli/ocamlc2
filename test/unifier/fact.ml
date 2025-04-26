
(* Recursive function *)
let rec factorial n =
  if n <= 1 then 1
  else n * factorial (n - 1);;

let x = factorial 10
    in (print_int x; print_endline "");;


(*
RUN: p3 %s --dump-types | FileCheck %s.ref
*)

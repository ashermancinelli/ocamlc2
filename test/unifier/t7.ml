type l = A | B of int | C of int * int;;
A;;
B 5;;
let bval : l = B 2 in
  let x = match bval with
    | A -> 0
    | _ -> 1
    in x
    ;;

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

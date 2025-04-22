type shape = A | B of int

let area (s : shape) : int =
    match s with
    | A -> 1
    | B i -> (i + 1)
    ;;

print_int (area (A));;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: area : (λ shape int)
*)

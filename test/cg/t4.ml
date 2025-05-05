type shape = A | B of int

let area (s : shape) : int =
    match s with
    | A -> 1
    | B i -> (i + 1)
    ;;

print_int (area A);
print_endline "";;

(*
RUN: g3 %s --dump-camlir | FileCheck %s
XFAIL: *
*)


let add1 = (+) 1;;
let show () = print_int (add1 2); print_endline "";;
let () = show ();;

(*
TODO: partial application of varargs
let partially_apply_varargs = (Printf.printf "Hello %s\n");;
let () = partially_apply_varargs "world";;
*)

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

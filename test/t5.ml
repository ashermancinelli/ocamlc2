let () = print_int (5 + 5);;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: () : unit
*)

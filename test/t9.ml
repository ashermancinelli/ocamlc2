let () = print_int (if true then 1 else if false then 2 else 3);;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: () : unit
*)

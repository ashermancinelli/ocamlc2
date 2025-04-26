let rec f x = if x > 5 then x else f (x + 1) in print_int (f 0);;

(*
RUN: p3 %s --dump-types | FileCheck %s.ref
*)

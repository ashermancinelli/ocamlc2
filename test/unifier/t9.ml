let f () = print_int (if true then 1 else if false then 2 else 3);;

let () = f ()

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

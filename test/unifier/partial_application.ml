let add1 = (+) 1;;
let show () = print_int (add1 2);;
let () = show ()

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: add1 : (λ int int)
CHECK: let: show : (λ unit unit)
*)

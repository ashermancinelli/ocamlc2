let x = (+);;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: x : (λ int int int)
*)

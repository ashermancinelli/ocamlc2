let add a = a + 5;;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: add : (λ int int)
*)

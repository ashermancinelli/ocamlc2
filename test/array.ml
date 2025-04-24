(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: iter : (λ (λ '[[T:.+]] unit) (array '[[T]]) unit)
*)
let iter = Array.iter;;
let len = Array.length;;

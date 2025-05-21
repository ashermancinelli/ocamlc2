
let f lb ub = 
    let x = lb in
    let y = ub in
        for i = x to y do
            print_int i
        done;;

(*
RUN: g3 %s --dump-camlir | FileCheck %s
CHECK-LABEL:   func.func private @f(
*)

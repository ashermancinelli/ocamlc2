
let f lb ub = 
    let x = lb in
    let y = ub in
        for i = x to y do
            print_int i
        done;;

(*
RUN: g3 %s | FileCheck %s.ref
*)


let f lb ub = 
    let x = lb in
    let y = ub in
        for i = x to y do
            print_int i
        done;;

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

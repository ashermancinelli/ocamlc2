let mean (s:int) (r:int) : int = (s + r) / 2;;

let x = 5 in
let y = 10 in
    print_int (mean x y);;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: mean : (λ int int int)
CHECK: let: x : int
CHECK: let: y : int
*)

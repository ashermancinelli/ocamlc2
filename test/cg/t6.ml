let mean (s:int) (r:int) : int = (s + r) / 2;;

let x = 5 in
let y = 10 in
    print_int (mean x y);;

(*
RUN: g3 %s
*)

let f ~a:int ~b ?(c:int = 0) = a + b + c;;
let g ?(a = 1) ?b c = a + c;;

(*
let () =
    let x = 1 in
    let y = 2 in
    let c = f ~a:x ~b:y in
    Printf.printf "c = %d\n" c;;

RUN: p3 --freestanding --dump-types %s | FileCheck %s
CHECK: let: f
CHECK: let: g
CHECK: need to update optional/default parameters
*)

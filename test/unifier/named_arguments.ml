let a (b : int) : int = b;;
let f ~a:a2 = a2;;
let g ?(b:int = 0) () = b;;
let h ?c x = match c with
    | Some c -> c
    | None -> x;;
let j ?(a = 0) () = a;;
let k ~(a:int) = a;;
let sink arg = ();;

let () =
    sink @@ f ~a:2;
    sink @@ g ~b:1 ();
    sink @@ g ();
    sink @@ h ~c:3 4;
    sink @@ h 1;
    sink @@ j ~a:4 ();
    sink @@ j ();
    sink @@ k ~a:5;;
(*
    Printf.printf "c = %d\n" c;;

RUN: p3 --freestanding --dump-types %s | FileCheck %s

https://github.com/tree-sitter/tree-sitter-ocaml/issues/118
XFAIL: *

CHECK: let: f
CHECK: let: g
CHECK: need to update optional/default parameters
*)

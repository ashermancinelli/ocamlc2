let a (b : int) : int = b;;
let f ~a:a2 = a2;;
let g ?(b:int = 0) () = b;;
let h ?c x = match c with
    | Some c -> c
    | None -> x;;
let j ?(a = 0) () = a;;
let k ~(a:int) = a;;

let () =
    print_int @@ f ~a:2;
    print_int @@ g ~b:1 ();
    print_int @@ g ();
    print_int @@ h ~c:3 4;
    print_int @@ h 1;
    print_int @@ j ~a:4 ();
    print_int @@ j ();
    print_int @@ k ~a:5;
    print_endline "";;
(*
    Printf.printf "c = %d\n" c;;

RUN: p3 --freestanding --dump-types %s | FileCheck %s

https://github.com/tree-sitter/tree-sitter-ocaml/issues/118
XFAIL: *

CHECK: let: f
CHECK: let: g
CHECK: need to update optional/default parameters
*)

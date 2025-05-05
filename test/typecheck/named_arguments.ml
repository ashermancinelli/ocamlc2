let a (b : int) : int = b;;
let f ~a:a2 = a2;;
let g ?(b:int = 0) () = b;;
let foo ?z:(z=0) x = x;;
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
    sink @@ foo ~z:1 2;
    sink @@ k ~a:5;;
(*
RUN: p3 --freestanding --dtypes %s | FileCheck %s.ref
*)

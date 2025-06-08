(*
RUN: g3 %s | FileCheck %s.ref
TODO: if we ever move to a real ast, we should do an early rewrite of
    optional args to use pattern matching and optionals. this would
    better reflect the semantics and then we don't have to create the
    pattern matching in an ad-hoc way during codegen, we can rely on
    the default lowering.
*)
type 'a option =
    | Some of 'a
    | None;;

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

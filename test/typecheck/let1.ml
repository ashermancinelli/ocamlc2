(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)

type 'a option = None | Some of 'a

let ( let* ) o f = match o with None -> None | Some x -> f x
let return x = Some x
let maybe () = Some 5
let maybe2 () = None

let z =
  let* x = maybe () in
  let* y = maybe2 () in
  return (x + y)

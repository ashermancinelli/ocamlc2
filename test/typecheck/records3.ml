(*
RUN: p3 -f -d %s | FileCheck %s.ref
*)

type 'a t = {x : 'a; y : 'a}
type ('a, 'b) t2 = {x : 'a; y : 'b; z : int}

let t : 'a t = {x = 1; y = 2}
let t2 : ('a, 'b) t2 = {x = 1; y = 2; z = 3}
let t3 = {x = 1; y = 2; z = 3}

let () = print_newline @@ print_int t.x
let () = print_newline @@ print_int t2.x;;

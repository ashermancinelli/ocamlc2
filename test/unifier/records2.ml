(*
RUN: p3 -d %S/mon.ml %s | FileCheck %s.ref
XFAIL: *
*)
open Mon
let c = {name = "Charmander"; hp = 39; ptype = TFire};;
print_int @@ match c with {name; hp; ptype} -> hp;;
print_endline @@ match c with {name = my_name; _; _} -> my_name

(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)
type ptype = TNormal | TFire | TWater
type mon = {name : string; hp : int; ptype : ptype}
let c = {name = "Charmander"; hp = 39; ptype = TFire};;
let x = match c with {name = my_name; hp = 50; ptype} -> my_name | _ -> "unknown";;
(* let y = match c with {name; hp; _} -> hp in 
let f (m:mon) = match m with {name; hp; _} -> hp in *)

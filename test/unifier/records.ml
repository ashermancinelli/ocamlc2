(*
RUN: p3 -f -d -cpp %s | FileCheck %s.ref
RUN: ocamlc %s -i -pp cppo
*)

module List : sig
  val filter : ('a -> bool) -> 'a list -> 'a list
#ifndef OCAMLC2
end = struct
  let filter f lst = ListLabels.filter ~f:f lst
#endif
end

type ptype = TNormal | TFire | TWater;;
type mon = {name : string; hp : int; ty : ptype};;

let mon1 = {name = "Bulbasaur"; hp = 100; ty = TNormal}
let mon2 = {name = "Charmander"; hp = 100; ty = TFire}
let mon3 = {name = "Squirtle"; hp = 100; ty = TWater}

let mon_list = [mon1; mon2; mon3]

let mon_list_of_type (ptype : ptype) : mon list =
  List.filter (fun m -> m.ty = ptype) mon_list

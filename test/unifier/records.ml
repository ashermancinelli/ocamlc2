(*
RUN: p3 --freestanding --dump-types %s | FileCheck %s.ref
XFAIL: *
*)
type ptype = TNormal | TFire | TWater
type mon = {name : string; hp : int; ptype : ptype}

let mon1 = {name = "Bulbasaur"; hp = 100; ptype = TNormal}
let mon2 = {name = "Charmander"; hp = 100; ptype = TFire}
let mon3 = {name = "Squirtle"; hp = 100; ptype = TWater}

let mon_list = [mon1; mon2; mon3]

let mon_list_of_type (ptype : ptype) : mon list =
  List.filter (fun mon -> mon.ptype = ptype) mon_list

let mon_list_of_type TNormal = mon_list_of_type TNormal


(*
RUN: p3 --dtypes %s | FileCheck %s.ref
*)
type ptype = TNormal | TFire | TWater
type mon = {name : string; hp : int; ty : ptype}

let mon1 = {name = "Bulbasaur"; hp = 100; ty = TNormal}
let mon2 = {name = "Charmander"; hp = 100; ty = TFire}
let mon3 = {name = "Squirtle"; hp = 100; ty = TWater}

let mon_list = [mon1; mon2; mon3]

let mon_list_of_type (ty : ptype) : mon list =
  List.filter (fun m -> m.ty = ty) mon_list

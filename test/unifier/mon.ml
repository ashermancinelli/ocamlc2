(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)
type ptype = TNormal | TFire | TWater
type mon = {name : string; hp : int; ptype : ptype}

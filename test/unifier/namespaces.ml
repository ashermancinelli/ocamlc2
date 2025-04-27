(* Types and variable names should live in different namespaces *)
(*
RUN: p3 -d %s | FileCheck %s.ref
XFAIL: *
*)

type x = float;;
let f (x:int) (y:x) : float = y +. (float_of_int x);;

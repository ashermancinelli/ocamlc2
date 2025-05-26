external (+) : int -> int -> int = "ocaml_add"
external print_int : int -> unit = "ocaml_print_int"

let add1 = (+) 1;;
let two = add1 1 in print_int two;

(*
 * RUN: g3 %s | FileCheck %s.ref
 *)

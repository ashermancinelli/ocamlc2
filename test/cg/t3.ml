
let f lb ub = 
    let x = lb in
    let y = ub in
        for i = x to y do
            print_int i
        done;;

(*
RUN: g3 %s --dump-camlir | FileCheck %s
// CHECK-LABEL:   func.func private @f(
// CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !ocaml.box<i64>,
// CHECK-SAME:                         %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !ocaml.box<i64>) -> !ocaml.unit {
*)

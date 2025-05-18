let test x =
  if x > 10 then
    "large"
  else if x > 5 then
    "medium"
  else if x > 0 then
    "small"
  else
    "negative or zero"

let () = print_string (test 7) 

(*
 * RUN: g3 %s | FileCheck %s
 * CHECK-LABEL:   func.func private @print_string(!ocaml.sbox) -> !ocaml.unit
 * CHECK:         func.func private @">"(!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i1>
 * CHECK-LABEL:   func.func private @test(
 * CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !ocaml.box<i64>) -> !ocaml.sbox {
 * CHECK:           %[[VAL_1:.*]] = arith.constant 10 : i64
 * CHECK:           %[[VAL_2:.*]] = ocaml.convert %[[VAL_1]] from i64 to !ocaml.box<i64>
 * CHECK:           %[[VAL_3:.*]] = call @">"(%[[VAL_0]], %[[VAL_2]]) : (!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i1>
 * CHECK:           %[[VAL_4:.*]] = ocaml.convert %[[VAL_3]] from !ocaml.box<i1> to i1
 * CHECK:           %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (!ocaml.sbox) {
 * CHECK:             %[[VAL_6:.*]] = ocaml.embox_string "\22large\22"
 * CHECK:             scf.yield %[[VAL_6]] : !ocaml.sbox
 * CHECK:           } else {
 * CHECK:             %[[VAL_7:.*]] = arith.constant 5 : i64
 * CHECK:             %[[VAL_8:.*]] = ocaml.convert %[[VAL_7]] from i64 to !ocaml.box<i64>
 * CHECK:             %[[VAL_9:.*]] = func.call @">"(%[[VAL_0]], %[[VAL_8]]) : (!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i1>
 * CHECK:             %[[VAL_10:.*]] = ocaml.convert %[[VAL_9]] from !ocaml.box<i1> to i1
 * CHECK:             %[[VAL_11:.*]] = scf.if %[[VAL_10]] -> (!ocaml.sbox) {
 * CHECK:               %[[VAL_12:.*]] = ocaml.embox_string "\22medium\22"
 * CHECK:               scf.yield %[[VAL_12]] : !ocaml.sbox
 * CHECK:             } else {
 * CHECK:               %[[VAL_13:.*]] = arith.constant 0 : i64
 * CHECK:               %[[VAL_14:.*]] = ocaml.convert %[[VAL_13]] from i64 to !ocaml.box<i64>
 * CHECK:               %[[VAL_15:.*]] = func.call @">"(%[[VAL_0]], %[[VAL_14]]) : (!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i1>
 * CHECK:               %[[VAL_16:.*]] = ocaml.convert %[[VAL_15]] from !ocaml.box<i1> to i1
 * CHECK:               %[[VAL_17:.*]] = scf.if %[[VAL_16]] -> (!ocaml.sbox) {
 * CHECK:                 %[[VAL_18:.*]] = ocaml.embox_string "\22small\22"
 * CHECK:                 scf.yield %[[VAL_18]] : !ocaml.sbox
 * CHECK:               } else {
 * CHECK:                 %[[VAL_19:.*]] = ocaml.embox_string "\22negative or zero\22"
 * CHECK:                 scf.yield %[[VAL_19]] : !ocaml.sbox
 * CHECK:               }
 * CHECK:               scf.yield %[[VAL_17]] : !ocaml.sbox
 * CHECK:             }
 * CHECK:             scf.yield %[[VAL_11]] : !ocaml.sbox
 * CHECK:           }
 * CHECK:           return %[[VAL_5]] : !ocaml.sbox
 * CHECK:         }
 * CHECK-LABEL:   func.func private @main() -> i32 {
 * CHECK:           %[[VAL_0:.*]] = ocaml.unit
 * CHECK:           %[[VAL_1:.*]] = arith.constant 7 : i64
 * CHECK:           %[[VAL_2:.*]] = ocaml.convert %[[VAL_1]] from i64 to !ocaml.box<i64>
 * CHECK:           %[[VAL_3:.*]] = call @test(%[[VAL_2]]) : (!ocaml.box<i64>) -> !ocaml.sbox
 * CHECK:           %[[VAL_4:.*]] = call @print_string(%[[VAL_3]]) : (!ocaml.sbox) -> !ocaml.unit
 * CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
 * CHECK:           return %[[VAL_5]] : i32
 * CHECK:         }
 *)

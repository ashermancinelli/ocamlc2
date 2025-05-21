(*
 * RUN: g3 %s | FileCheck %s
 *)
let x = 5 in
let y = 7 in
let z () = x + y in
z ();;
(*
  * CHECK-LABEL:   func.func private @"+"(!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i64>

 * CHECK-LABEL:   func.func private @z(
 * CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !ocaml.unit) -> !ocaml.box<i64> attributes {env = "zenv$0"} {
 * CHECK:           %[[VAL_1:.*]] = ocaml.closure.env.get_current : !ocaml.env
 * CHECK:           %[[VAL_2:.*]] = ocaml.closure.env.get %[[VAL_1]]["y"] -> !ocaml.box<i64>
 * CHECK:           %[[VAL_3:.*]] = ocaml.closure.env.get_current : !ocaml.env
 * CHECK:           %[[VAL_4:.*]] = ocaml.closure.env.get %[[VAL_3]]["x"] -> !ocaml.box<i64>
 * CHECK:           %[[VAL_5:.*]] = call @"+"(%[[VAL_4]], %[[VAL_2]]) : (!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i64>
 * CHECK:           return %[[VAL_5]] : !ocaml.box<i64>
 * CHECK:         }

 * CHECK-LABEL:   func.func private @main() -> i32 {
 * CHECK:           %[[VAL_0:.*]] = scf.execute_region -> !ocaml.box<i64> {
 * CHECK:             %[[VAL_1:.*]] = arith.constant 5 : i64
 * CHECK:             %[[VAL_2:.*]] = ocaml.convert %[[VAL_1]] from i64 to !ocaml.box<i64>
 * CHECK:             %[[VAL_3:.*]] = scf.execute_region -> !ocaml.box<i64> {
 * CHECK:               %[[VAL_4:.*]] = arith.constant 7 : i64
 * CHECK:               %[[VAL_5:.*]] = ocaml.convert %[[VAL_4]] from i64 to !ocaml.box<i64>
 * CHECK:               %[[VAL_6:.*]] = scf.execute_region -> !ocaml.box<i64> {
 * CHECK:                 %[[VAL_7:.*]] = ocaml.closure.env.new {for = "zenv$0"}
 * CHECK:                 ocaml.closure.env.capture %[[VAL_7]]["y"] = %[[VAL_5]] : !ocaml.box<i64>
 * CHECK:                 ocaml.closure.env.capture %[[VAL_7]]["x"] = %[[VAL_2]] : !ocaml.box<i64>
 * CHECK:                 %[[VAL_8:.*]] = ocaml.unit
 * CHECK:                 %[[VAL_9:.*]] = ocaml.unit
 * CHECK:                 %[[VAL_10:.*]] = func.call @z(%[[VAL_9]]) : (!ocaml.unit) -> !ocaml.box<i64>
 * CHECK:                 scf.yield %[[VAL_10]] : !ocaml.box<i64>
 * CHECK:               }
 * CHECK:               scf.yield %[[VAL_6]] : !ocaml.box<i64>
 * CHECK:             }
 * CHECK:             scf.yield %[[VAL_3]] : !ocaml.box<i64>
 * CHECK:           }
 * CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
 * CHECK:           return %[[VAL_11]] : i32
 * CHECK:         }

 *)

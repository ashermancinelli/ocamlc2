let () =
  for i = 1 to 10 do
    print_float (float_of_int i)
  done

let foo () = 1 + 2;;

(*
RUN: g3 -dump-camlir %s | FileCheck %s
// CHECK-LABEL:   func.func private @main() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = ocaml.convert %[[VAL_0]] from i64 to !ocaml.box<i64>
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_3:.*]] = ocaml.convert %[[VAL_2]] from i64 to !ocaml.box<i64>
// CHECK:           %[[VAL_4:.*]] = ocaml.convert %[[VAL_1]] from !ocaml.box<i64> to i64
// CHECK:           %[[VAL_5:.*]] = ocaml.convert %[[VAL_3]] from !ocaml.box<i64> to i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]]  : i64 {
// CHECK:             %[[VAL_8:.*]] = ocaml.convert %[[VAL_7]] from i64 to !ocaml.box<i64>
// CHECK:             %[[VAL_9:.*]] = func.call @float_of_int(%[[VAL_8]]) : (!ocaml.box<i64>) -> !ocaml.box<f64>
// CHECK:             %[[VAL_10:.*]] = func.call @print_float(%[[VAL_9]]) : (!ocaml.box<f64>) -> !ocaml.unit
// CHECK:             %[[VAL_11:.*]] = ocaml.unit
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = constant @foo : (!ocaml.unit) -> !ocaml.box<i64>
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_13]] : i32
// CHECK:         }
*)

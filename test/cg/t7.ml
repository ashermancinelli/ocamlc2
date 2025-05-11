type l = A | B of int | C of int * int;;
A;;
B 5;;
let bval : l = B 2 in
  let x = match bval with
    | A -> 0
    | _ -> 1
    in x
    ;;

(*
RUN: g3 %s
*)

(*
 * CHECK-LABEL:   func.func private @B(!ocaml.box<i64>) -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>

 * CHECK-LABEL:   func.func private @A() -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>> attributes {ocaml.variant_ctor} {
 * CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
 * CHECK:           %[[VAL_1:.*]] = ocaml.builtin "variant_ctor_empty"(%[[VAL_0]]) : i64 -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:           return %[[VAL_1]] : !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:         }

 * CHECK-LABEL:   func.func private @main() -> i32 {
 * CHECK:           %[[VAL_0:.*]] = call @A() : () -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:           %[[VAL_1:.*]] = arith.constant 5 : i64
 * CHECK:           %[[VAL_2:.*]] = ocaml.convert %[[VAL_1]] from i64 to !ocaml.box<i64>
 * CHECK:           %[[VAL_3:.*]] = call @B(%[[VAL_2]]) : (!ocaml.box<i64>) -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:           %[[VAL_4:.*]] = scf.execute_region -> !ocaml.box<i64> {
 * CHECK:             %[[VAL_5:.*]] = arith.constant 2 : i64
 * CHECK:             %[[VAL_6:.*]] = ocaml.convert %[[VAL_5]] from i64 to !ocaml.box<i64>
 * CHECK:             %[[VAL_7:.*]] = func.call @B(%[[VAL_6]]) : (!ocaml.box<i64>) -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:             %[[VAL_8:.*]] = scf.execute_region -> !ocaml.box<i64> {
 * CHECK:               %[[VAL_9:.*]] = scf.execute_region -> !ocaml.box<i64> {
 * CHECK:                 %[[VAL_10:.*]] = ocaml.pattern_variable : !ocaml.obox
 * CHECK:                 %[[VAL_11:.*]] = ocaml.convert %[[VAL_10]] from !ocaml.obox to !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:                 %[[VAL_12:.*]] = ocaml.match %[[VAL_7]] against %[[VAL_11]] : !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:                 cf.cond_br %[[VAL_12]], ^bb1, ^bb2
 * CHECK:               ^bb1:
 * CHECK:                 %[[VAL_13:.*]] = arith.constant 1 : i64
 * CHECK:                 %[[VAL_14:.*]] = ocaml.convert %[[VAL_13]] from i64 to !ocaml.box<i64>
 * CHECK:                 cf.br ^bb5(%[[VAL_14]] : !ocaml.box<i64>)
 * CHECK:               ^bb2:
 * CHECK:                 %[[VAL_15:.*]] = func.call @A() : () -> !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:                 %[[VAL_16:.*]] = ocaml.match %[[VAL_7]] against %[[VAL_15]] : !ocaml.variant<"l" is "A" | "B" of !ocaml.box<i64> | "C" of tuple<!ocaml.box<i64>, !ocaml.box<i64>>>
 * CHECK:                 cf.cond_br %[[VAL_16]], ^bb3, ^bb4
 * CHECK:               ^bb3:
 * CHECK:                 %[[VAL_17:.*]] = arith.constant 0 : i64
 * CHECK:                 %[[VAL_18:.*]] = ocaml.convert %[[VAL_17]] from i64 to !ocaml.box<i64>
 * CHECK:                 cf.br ^bb5(%[[VAL_18]] : !ocaml.box<i64>)
 * CHECK:               ^bb4:
 * CHECK:                 %[[VAL_19:.*]] = arith.constant false
 * CHECK:                 cf.assert %[[VAL_19]], "No match found"
 * CHECK:                 %[[VAL_20:.*]] = ocaml.convert %[[VAL_19]] from i1 to !ocaml.box<i64>
 * CHECK:                 cf.br ^bb5(%[[VAL_20]] : !ocaml.box<i64>)
 * CHECK:               ^bb5(%[[VAL_21:.*]]: !ocaml.box<i64>):
 * CHECK:                 scf.yield %[[VAL_21]] : !ocaml.box<i64>
 * CHECK:               }
 * CHECK:               scf.yield %[[VAL_9]] : !ocaml.box<i64>
 * CHECK:             }
 * CHECK:             scf.yield %[[VAL_8]] : !ocaml.box<i64>
 * CHECK:           }
 * CHECK:           %[[VAL_22:.*]] = arith.constant 0 : i32
 * CHECK:           return %[[VAL_22]] : i32
 * CHECK:         }
 *)

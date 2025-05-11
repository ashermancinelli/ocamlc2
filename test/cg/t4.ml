type shape = A | B of int

let area (s : shape) : int =
    match s with
    | A -> 1
    | B i -> (i + 1)
    ;;

print_int (area A);
print_endline "";;

(*
RUN: g3 %s --dump-camlir | FileCheck %s
CHECK-LABEL:   func.func private @main() -> i32 {
CHECK:           %[[VAL_0:.*]] = ocaml.unit
CHECK:           %[[VAL_1:.*]] = call @A() : () -> !ocaml.variant<"shape" is "A" | "B" of !ocaml.box<i64>>
CHECK:           %[[VAL_2:.*]] = call @area(%[[VAL_1]]) : (!ocaml.variant<"shape" is "A" | "B" of !ocaml.box<i64>>) -> !ocaml.box<i64>
CHECK:           %[[VAL_3:.*]] = call @print_int(%[[VAL_2]]) : (!ocaml.box<i64>) -> !ocaml.unit
CHECK:           %[[VAL_4:.*]] = ocaml.embox_string "\22\22"
CHECK:           %[[VAL_5:.*]] = call @print_endline(%[[VAL_4]]) : (!ocaml.sbox) -> !ocaml.unit
CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
CHECK:           return %[[VAL_6]] : i32
CHECK:         }
*)

let message = Moda.message;;
(*
RUN: p3 %S/moda.ml %s --dtypes | FileCheck %s
CHECK-COUNT-2: let: message : string
*)

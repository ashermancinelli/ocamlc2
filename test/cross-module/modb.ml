let message = Moda.message;;
(*
RUN: p3 %S/moda.ml %s --dump-types | FileCheck %s
CHECK-COUNT-2: let: message : string
*)

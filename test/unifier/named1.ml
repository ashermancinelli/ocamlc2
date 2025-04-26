let a ?(b : int) () = b;;
(*
RUN: p3 %s --dump-types | FileCheck %s.ref
*)

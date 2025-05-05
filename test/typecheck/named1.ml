let a ?(b : int) () = b;;
(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

let f x = let two () = x + 2 in two ();;
f 5;

(**
 * RUN: g3 %s | FileCheck %s.ref
 *)

(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)

(* Define a signature for the functor argument *)
module type X_int = sig
  val x : int
end

(* Define a functor that takes a module matching X_int *)
module F (M : X_int) = struct
  let y = M.x + 1
end

(* Define a module that matches X_int *)
module A = struct
  let x = 10
end

(* Apply the functor *)
module B = F (A)

(* Use a value from the resulting module *)
let z = B.y

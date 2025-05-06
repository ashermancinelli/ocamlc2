
(* 
RUN: p3 -d -f %s | FileCheck %s.ref
*)

module type SigA = sig
  val x : int
end

module type SigB = sig
  val y : string
end

(* Define the functor that takes two modules *)
module Make (A : SigA) (B : SigB) = struct
  let combined = string_of_int A.x ^ B.y
end

(* Define concrete modules matching the signatures *)
module ConcreteA = struct
  let x = 42
end

module ConcreteB = struct
  let y = " world"
end

(* Instantiate the functor with the concrete modules *)
module CombinedModule = Make (ConcreteA) (ConcreteB)

(* Example of using the resulting module *)
let result = CombinedModule.combined

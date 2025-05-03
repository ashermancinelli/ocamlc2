(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)

(* Signature for the functor's input *)
module type INPUT = sig
  val input_val : int
end

(* Signature for the functor's explicit output *)
module type OUTPUT = sig
  val output_val : bool
end

(* Functor definition with explicit result signature OUTPUT *)
module F (M : INPUT) : OUTPUT = struct
  let output_val = (M.input_val > 0)
end

(* Argument module matching INPUT *)
module ArgImpl = struct
  let input_val = 5
  let extra_decl = "test";
end

(* Apply the functor *)
module Result = F (ArgImpl)

(* Use the result *)
let final_result = Result.output_val;;

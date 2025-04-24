(* Test file for Hindley-Milner type inference *)

(* Polymorphic identity function *)
let id x = x;;

(* Apply identity to different types *)
let a = id 42;;
let b = id true;;

(* Function with type annotations *)
let add (x : int) (y : int) = x + y;;

(* Higher-order functions *)
let apply f x = f x;;
let compose f g x = f (g x);;

(* Function application *)
let result = compose (fun x -> x * 2) (fun y -> y + 1) 5;; 

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: id : (λ '[[T:.+]] '[[T]])
CHECK: let: a : int
CHECK: let: b : bool
CHECK: let: add : (λ int int int)
CHECK: let: apply : (λ (λ '[[T1:.+]] '[[T2:.+]]) '[[T1]] '[[T2]])
CHECK: let: compose : (λ (λ '[[T3:.+]] '[[T4:.+]]) (λ '[[T5:.+]] '[[T3]]) '[[T5]] '[[T4]])
CHECK: let: result : int
*)

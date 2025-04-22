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
*)

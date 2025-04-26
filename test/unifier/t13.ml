(* Simple expressions *)
let x = 42;;
let y = x + 5;;

(* Function definition *)
let add a b = a + b;;

(* Function application *)
let z = add 10 20;;

(* Polymorphic identity function *)
let id x = x;;

(* Using the identity function with different types *)
let a = id 42;;
let b = id true;;

(* Higher-order function *)
let apply f x = f x;;

(* Composition *)
let compose f g x = f (g x);;

(* Recursive function *)
let rec factorial n = 
  if n <= 1 then 1 else n * factorial (n - 1);;

(* Pattern matching *)
type shape = Circle | Rectangle | Triangle;;

let describe_shape s = 
  match s with
  | Circle -> "circle"
  | Rectangle -> "rectangle" 
  | Triangle -> "triangle";;

(* Last expression to test *)
let result = compose (fun x -> x * 2) (fun y -> y + 1) 5;; 

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

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
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: x : int
CHECK: let: y : int
CHECK: let: add : (λ int int int)
CHECK: let: z : int
CHECK: let: id : (λ 't17 't17)
CHECK: let: a : int
CHECK: let: b : bool
CHECK: let: apply : (λ (λ 't23 't24) 't23 't24)
CHECK: let: compose : (λ (λ 't28 't29) (λ 't27 't28) 't27 't29)
CHECK: let: factorial : (λ int int)
CHECK: let: describe_shape : (λ shape string)
CHECK: let: result : int
*)

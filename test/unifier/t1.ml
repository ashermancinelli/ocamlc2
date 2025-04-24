(* Basic variable binding *)
let x = 42
let greeting = "Hello, OCaml!"

(* Function definition *)
let add a b = a + b

(* Recursive function *)
let rec factorial n =
  if n <= 1 then 1
  else n * factorial (n - 1)

(* Pattern matching *)
let describe_number n =
  match n with
  | 0 -> "Zero"
  | 1 -> "One"
  | 2 -> "Two"
  | n when n > 0 -> "Positive"
  | _ -> "Negative"

(* Option type and handling *)
let find_index item arr =
  let rec aux i =
    if i >= Array.length arr then None
    else if arr.(i) = item then Some i
    else aux (i + 1)
  in
  aux 0

(* List operations *)
let my_list = [1; 2; 3; 4; 5]
let doubled = List.map (fun x -> x * 2) my_list
let sum = List.fold_left (+) 0 my_list

(* Custom type definition *)
type shape =
  | Circle of float  (* radius *)
  | Rectangle of float * float  (* width, height *)
  | Triangle of float * float * float  (* sides *)

(* Function using custom type *)
let area s = match s with
  | Circle r -> Float.pi *. r *. r
  | Rectangle (w, h) -> w *. h
  | Triangle (a, b, c) ->
      let s = (a +. b +. c) /. 2.0 in
      sqrt (s *. (s -. a) *. (s -. b) *. (s -. c))

(* Print results to verify *)
let () =
  Printf.printf "Value of x: %d\n" x;
  Printf.printf "Greeting: %s\n" greeting;
  Printf.printf "5 + 10 = %d\n" (add 5 10);
  Printf.printf "Factorial of 5: %d\n" (factorial 5);
  Printf.printf "Describe 0: %s\n" (describe_number 0);
  Printf.printf "Describe 5: %s\n" (describe_number 5);
  
  let test_array = [|10; 20; 30; 40; 50|] in
  (match find_index 30 test_array with
  | Some i -> Printf.printf "Found 30 at index: %d\n" i
  | None -> Printf.printf "30 not found\n");
  
  Printf.printf "Original list: [%s]\n" (String.concat "; " (List.map string_of_int my_list));
  Printf.printf "Doubled list: [%s]\n" (String.concat "; " (List.map string_of_int doubled));
  Printf.printf "Sum of list: %d\n" sum;
  
  Printf.printf "Area of circle (r=5): %f\n" (area (Circle 5.0));
  Printf.printf "Area of rectangle (3x4): %f\n" (area (Rectangle (3.0, 4.0)));
  Printf.printf "Area of triangle (3,4,5): %f\n" (area (Triangle (3.0, 4.0, 5.0)));

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: x : int
CHECK: let: greeting : string
CHECK: let: add : (λ int int int)
CHECK: let: factorial : (λ int int)
CHECK: let: describe_number : (λ int string)
CHECK: let: aux : (λ int (Optional int))
CHECK: let: find_index : (λ '[[T:.+]] (Array '[[T]]) (Optional int))
CHECK: let: my_list : (List int)
CHECK: let: doubled : (List int)
CHECK: let: sum : int
CHECK: let: s : float
CHECK: let: area : (λ shape float)
CHECK: let: test_array : (Array int)
CHECK: let: () : unit
*)

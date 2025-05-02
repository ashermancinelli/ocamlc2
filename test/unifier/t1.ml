(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

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
  print_string "Value of x: "; print_int x; print_newline ();
  print_string "Greeting: "; print_endline greeting;
  print_string "5 + 10 = "; print_int (add 5 10); print_newline ();
  print_string "Factorial of 5: "; print_int (factorial 5); print_newline ();
  print_string "Describe 0: "; print_endline (describe_number 0);
  print_string "Describe 5: "; print_endline (describe_number 5);
  
  let test_array = [|10; 20; 30; 40; 50|] in
  (match find_index 30 test_array with
  | Some i -> print_string "Found 30 at index: "; print_int i; print_newline ()
  | None -> print_endline "30 not found");
  
  print_string "Original list: [";
  print_string (String.concat "; " (List.map string_of_int my_list));
  print_endline "]";
  
  print_string "Doubled list: [";
  print_string (String.concat "; " (List.map string_of_int doubled));
  print_endline "]";
  
  print_string "Sum of list: "; print_int sum; print_newline ();
  
  print_string "Area of circle (r=5): ";
  print_float (area (Circle 5.0));
  print_newline ();
  
  print_string "Area of rectangle (3x4): ";
  print_float (area (Rectangle (3.0, 4.0)));
  print_newline ();
  
  print_string "Area of triangle (3,4,5): ";
  print_float (area (Triangle (3.0, 4.0, 5.0)));
  print_newline ();

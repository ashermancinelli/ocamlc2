let find_index item arr =
  let rec aux i =
    if i >= Array.length arr then (None)
    else if arr.(i) = item then Some i
    else aux (i + 1)
  in
  aux 0

let () =
  let arr = [|1; 2; 3; 4; 5|] in
  let index = find_index 3 arr in
  match index with
  | Some i -> print_string "Index of 3: "; print_int i; print_newline ()
  | None -> print_string "3 not found in array"; print_newline ()

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

(* Option type and handling *)
let find_index item arr =
  let rec aux i =
    if i >= Array.length arr then None
    else if arr.(i) = item then Some i
    else aux (i + 1)
  in
  aux 0

let () =
  let arr = [|1; 2; 3; 4; 5|] in
  let item = 3 in
  let result = find_index item arr in
  print_endline (match result with Some i -> string_of_int i | None -> "Not found");

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

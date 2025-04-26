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
  | Some i -> Printf.printf "Index of 3: %d\n" i
  | None -> Printf.printf "3 not found in array\n"

(*
RUN: p3 %s --dump-types | FileCheck %s.ref
*)

module Array = struct
  external length : 'a array -> int = "caml_array_length"
  external get : 'a array -> int -> 'a = "caml_array_get"
end

external (=) : 'a -> 'a -> bool = "caml_equal"
external (>) : int -> int -> bool = "caml_int_gt"
external (>=) : int -> int -> bool = "caml_int_ge"
external print_string : string -> unit = "caml_print_string"
external print_int : int -> unit = "caml_print_int"
external print_newline : unit -> unit = "caml_print_newline"

let find_index item arr =
  let rec aux i =
    if i >= Array.length arr then (None)
    else if (Array.get arr i) = item then Some i
    else aux (i + 1)
  in
  aux 0

let () =
  let arr = [|1; 2; 3; 4; 5|] in
  let index = find_index 3 arr in
  match index with
  | Some i -> print_string "Index of 3: "; print_int i; print_newline ()
  | None -> print_string "3 not found in array"; print_newline ()

(**
 * RUN: g3 %s | FileCheck %s.ref
 *)

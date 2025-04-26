module M2 : sig
  val print : unit -> unit
  val map : ('a -> 'b) -> 'a list -> 'b list
end = struct
  let message = "Hello from M2"
  let print () = print_endline message
  let map f l = List.map f l
end

let () =
  M2.print ();
  let l = [1; 2; 3] in
  let l' = M2.map (fun x -> x * 2) l in
  let f x = print_endline @@ string_of_int x in
  (List.iter f l' ; List.iter (fun x -> print_endline "---") l');

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

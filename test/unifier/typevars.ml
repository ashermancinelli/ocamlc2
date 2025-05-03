
(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

module M = struct
  type colour =
    | Red | Green of int * int | Blue of float
    | RGB of {r : int; g : int; b : int}

  let () = print_endline @@ match Red with
    | Red -> "Red"
    | Blue 5.0 -> "Blue"
    | Green (a, b) -> "Green"
    | _ -> "whatever";;

  type 'a tree = Lf | Br of 'a * 'a tree * 'a;;
end;;



let () = 
  let f t = match t with
    | M.Br (a, b, c) -> "Br"
    | M.Lf -> "Lf"
  in
  print_endline @@ f (M.Br (1, M.Br (2, M.Lf, 4), 4));
;;

open M;;

let () = print_endline @@ match M.Br (1, M.Br (2, M.Lf, 4), 4) with
  | M.Br (a, b, c) -> "Br"
  | M.Lf -> "Lf"
;;

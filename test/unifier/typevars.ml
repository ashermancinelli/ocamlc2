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

let () = print_endline @@ match Br (1, Br (2, Lf, 4), 4) with
  | Br (a, b, c) -> "Br"
  | Lf -> "Lf"
;;

(*
RUN: p3 %s --dump-types | FileCheck %s
CHECK: let: f : (λ (tree '[[T:.+]]) string)
*)

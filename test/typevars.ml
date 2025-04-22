type colour =
  | Red | Green of int * int | Blue of float
  | RGB of {r : int; g : int; b : int}

let () = print_endline @@ match Red with
  | Red -> "Red"
  | Blue 5.0 -> "Blue"
  | Green (a, b) -> "Green"
  | _ -> "whatever";;

type 'a tree = Lf | Br of 'a * 'a tree * 'a;;

(*
type t = {decoration : string; substance : t'}
and t' = Int of int | List of t list
*)

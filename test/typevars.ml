type colour =
  | Red | Green of int * int | Blue of float * int
  | RGB of {r : int; g : int; b : int}

let () = print_endline @@ match Red with
  | Red -> "Red"
  | Green (a, b) -> "Green"
  | _ -> "whatever";;
type 'a tree = Lf | Br of 'a * 'a tree * 'a;;

(*
type t = {decoration : string; substance : t'}
and t' = Int of int | List of t list
*)

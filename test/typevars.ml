type colour =
  | Red | Green | Blue | Yellow | Black | White
  | RGB of {r : int; g : int; b : int}

type 'a tree = Lf | Br of 'a * 'a tree * 'a;;

type t = {decoration : string; substance : t'}
and t' = Int of int | List of t list

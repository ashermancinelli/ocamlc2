type l = A | B of int | C of int * int;;
A;;
B 5;;
let (x:int) = match B 5 with
  | A -> 0
  | B x -> x
  | C _ -> 1
  in print_int x
;;



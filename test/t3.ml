(* Printf.printf "%d\n" (Obj.magic (Obj.repr 5) : int);; *)
let p (i : int) : unit = print_int (i + 5 * 3);;

let x = 0 in
let y = 10 in
for i = x to y do
  p i
done;;


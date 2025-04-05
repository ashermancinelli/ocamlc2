(* Printf.printf "%d\n" (Obj.magic (Obj.repr 5) : int);; *)
let p (i : int) : unit = print_int i;;

for i = 0 to 5 do
  p i
done;;


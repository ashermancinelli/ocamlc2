(* Printf.printf "%d\n" (Obj.magic (Obj.repr 5) : int);; *)
let p (i : int) : unit = print_int i;;

let x = 0 in
for i = x to 5 do
  p i
done;;


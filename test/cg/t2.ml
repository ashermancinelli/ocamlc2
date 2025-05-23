(*
RUN: g3 %s | FileCheck %s.ref
*)
let () =
  for i = 1 to 10 do
    print_float (float_of_int i)
  done

let foo () = 1 + 2;;


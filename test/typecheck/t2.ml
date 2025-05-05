(* Print numbers 1 through 10 *)
let () =
  for i = 1 to 10 do
    print_float (float_of_int i)
  done

let foo () = 1 + 2;;

(*
RUN: p3 %s --dtypes | FileCheck %s.ref
*)

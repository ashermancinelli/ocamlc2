(*
 * RUN: g3 %s | FileCheck %s.ref
 *)
let make_adders () =
  let result = ref [] in
  for i = 1 to 3 do
    let f = fun x -> x + i in
    result := f :: !result
  done;
  !result

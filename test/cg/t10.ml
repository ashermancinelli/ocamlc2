let test x =
  if x > 10 then
    "large"
  else if x > 5 then
    "medium"
  else if x > 0 then
    "small"
  else
    "negative or zero"

let () = print_string (test 7) 

(*
 * RUN: g3 %s | FileCheck %s.ref
 *)

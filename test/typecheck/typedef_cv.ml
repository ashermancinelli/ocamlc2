(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
*)
module Seq = struct
  type 'a t = unit -> 'a node
  and +'a node = Nil | Cons of 'a * 'a t
end



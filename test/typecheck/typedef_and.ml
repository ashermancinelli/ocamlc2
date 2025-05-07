(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)
module Seq = struct
  type 'a t = unit -> 'a node
  and 'a node = Nil | Cons of 'a * 'a t
end

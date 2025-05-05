(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)

module type S = sig
  type t
end

module F(P:S) = struct
  type t = P.t
end

module M = F(struct
  type t = int
end)

module N = F(struct
  type t = float
end)

let f x:M.t = x;;
let g x:N.t = x;;

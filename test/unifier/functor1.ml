(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
*)

module type S = sig
  type t
end

module M : S = struct
  type t = float
end

module N : S = struct
  include M
end

module O : S = struct
  type t = int
end

let f x:M.t = x;;
let g x:N.t = x;;
let h x:O.t = x;;


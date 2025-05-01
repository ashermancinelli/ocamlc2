(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
*)

module M : sig
  type t
end = struct
  type t = int
end

module F = struct
  include M
end

let f x:F.t = x;;

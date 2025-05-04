(*
RUN: p3 -d -f %s | FileCheck %s.ref
*)

module type S = sig
  type t
end

module M : S = struct
  type t = float
end

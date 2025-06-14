
type 'a t = 'a array
external length : 'a array -> int = "%array_length"
external get : 'a array -> int -> 'a = "%array_safe_get"
external set : 'a array -> int -> 'a -> unit = "%array_safe_set"
external make : int -> 'a -> 'a array = "caml_array_make"
external create_float: int -> float array = "caml_array_create_float"
val init : int -> (int -> 'a) -> 'a array
val make_matrix : int -> int -> 'a -> 'a array array
val init_matrix : int -> int -> (int -> int -> 'a) -> 'a array array
val append : 'a array -> 'a array -> 'a array
val concat : 'a array list -> 'a array
val sub : 'a array -> int -> int -> 'a array
val copy : 'a array -> 'a array
val fill : 'a array -> int -> int -> 'a -> unit
val blit : 'a array -> int -> 'a array -> int -> int -> unit
val to_list : 'a array -> 'a list
val of_list : 'a list -> 'a array
val equal : ('a -> 'a -> bool) -> 'a array -> 'a array -> bool
val compare : ('a -> 'a -> int) -> 'a array -> 'a array -> int
val iter : ('a -> unit) -> 'a array -> unit
val iteri : (int -> 'a -> unit) -> 'a array -> unit
val map : ('a -> 'b) -> 'a array -> 'b array
val map_inplace : ('a -> 'a) -> 'a array -> unit
val mapi : (int -> 'a -> 'b) -> 'a array -> 'b array
val mapi_inplace : (int -> 'a -> 'a) -> 'a array -> unit
val fold_left : ('acc -> 'a -> 'acc) -> 'acc -> 'a array -> 'acc
val fold_left_map : ('acc -> 'a -> 'acc * 'b) -> 'acc -> 'a array -> 'acc * 'b array
val fold_right : ('a -> 'acc -> 'acc) -> 'a array -> 'acc -> 'acc
val iter2 : ('a -> 'b -> unit) -> 'a array -> 'b array -> unit
val iter2 : ('a -> 'b -> unit) -> 'a array -> 'b array -> unit
val map2 : ('a -> 'b -> 'c) -> 'a array -> 'b array -> 'c array
val for_all : ('a -> bool) -> 'a array -> bool
val exists : ('a -> bool) -> 'a array -> bool
val for_all2 : ('a -> 'b -> bool) -> 'a array -> 'b array -> bool
val exists2 : ('a -> 'b -> bool) -> 'a array -> 'b array -> bool
val mem : 'a -> 'a array -> bool
val memq : 'a -> 'a array -> bool
val find_opt : ('a -> bool) -> 'a array -> 'a option
val find_index : ('a -> bool) -> 'a array -> int option
val find_map : ('a -> 'b option) -> 'a array -> 'b option
val find_mapi : (int -> 'a -> 'b option) -> 'a array -> 'b option
val split : ('a * 'b) array -> 'a array * 'b array
val combine : 'a array -> 'b array -> ('a * 'b) array
val sort : ('a -> 'a -> int) -> 'a array -> unit
val stable_sort : ('a -> 'a -> int) -> 'a array -> unit
val fast_sort : ('a -> 'a -> int) -> 'a array -> unit
(*
RUN: p3 -d %s | FileCheck %s.ref
*)

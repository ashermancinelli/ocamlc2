
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
RUN: p3 %s --dump-types | FileCheck %s
CHECK: val: length : (λ (array '[[T1:.+]]) int)
CHECK: val: get : (λ (array '[[T1:.+]]) int '[[T1]])
CHECK: val: set : (λ (array '[[T1:.+]]) int '[[T1]] unit)
CHECK: val: make : (λ int '[[T1:.+]] (array '[[T1]]))
CHECK: val: create_float : (λ int (array float))
CHECK: val: init : (λ int (λ int '[[T1:.+]]) (array '[[T1]]))
CHECK: val: make_matrix : (λ int int '[[T1:.+]] (array (array '[[T1]])))
CHECK: val: init_matrix : (λ int int (λ int int '[[T1:.+]]) (array (array '[[T1]])))
CHECK: val: append : (λ (array '[[T1:.+]]) (array '[[T1]]) (array '[[T1]]))
CHECK: val: concat : (λ (list (array '[[T1:.+]])) (array '[[T1]]))
CHECK: val: sub : (λ (array '[[T1:.+]]) int int (array '[[T1]]))
CHECK: val: copy : (λ (array '[[T1:.+]]) (array '[[T1]]))
CHECK: val: fill : (λ (array '[[T1:.+]]) int int '[[T1]] unit)
CHECK: val: blit : (λ (array '[[T1:.+]]) int (array '[[T1]]) int int unit)
CHECK: val: to_list : (λ (array '[[T1:.+]]) (list '[[T1]]))
CHECK: val: of_list : (λ (list '[[T1:.+]]) (array '[[T1]]))
CHECK: val: equal : (λ (λ '[[T1:.+]] '[[T1]] bool) (array '[[T1]]) (array '[[T1]]) bool)
CHECK: val: compare : (λ (λ '[[T1:.+]] '[[T1]] int) (array '[[T1]]) (array '[[T1]]) int)
CHECK: val: iter : (λ (λ '[[T1:.+]] unit) (array '[[T1]]) unit)
CHECK: val: iteri : (λ (λ int '[[T1:.+]] unit) (array '[[T1]]) unit)
CHECK: val: map : (λ (λ '[[T1:.+]] '[[T2:.+]]) (array '[[T1]]) (array '[[T2]]))
CHECK: val: map_inplace : (λ (λ '[[T1:.+]] '[[T1]]) (array '[[T1]]) unit)
CHECK: val: mapi : (λ (λ int '[[T1:.+]] '[[T2:.+]]) (array '[[T1]]) (array '[[T2]]))
CHECK: val: mapi_inplace : (λ (λ int '[[T1:.+]] '[[T1]]) (array '[[T1]]) unit)
CHECK: val: fold_left : (λ (λ '[[T2:.+]] '[[T1:.+]] '[[T2]]) '[[T2]] (array '[[T1]]) '[[T2]])
CHECK: val: fold_left_map : (λ (λ '[[T2:.+]] '[[T1:.+]] (* '[[T2]] '[[T3:.+]])) '[[T2]] (array '[[T1]]) (* '[[T2]] (array '[[T3]])))
CHECK: val: fold_right : (λ (λ '[[T1:.+]] '[[T2:.+]] '[[T2]]) (array '[[T1]]) '[[T2]] '[[T2]])
CHECK: val: iter2 : (λ (λ '[[T1:.+]] '[[T2:.+]] unit) (array '[[T1]]) (array '[[T2]]) unit)
CHECK: val: map2 : (λ (λ '[[T1:.+]] '[[T2:.+]] '[[T3:.+]]) (array '[[T1]]) (array '[[T2]]) (array '[[T3]]))
CHECK: val: for_all : (λ (λ '[[T1:.+]] bool) (array '[[T1]]) bool)
CHECK: val: exists : (λ (λ '[[T1:.+]] bool) (array '[[T1]]) bool)
CHECK: val: for_all2 : (λ (λ '[[T1:.+]] '[[T2:.+]] bool) (array '[[T1]]) (array '[[T2]]) bool)
CHECK: val: exists2 : (λ (λ '[[T1:.+]] '[[T2:.+]] bool) (array '[[T1]]) (array '[[T2]]) bool)
CHECK: val: mem : (λ '[[T1:.+]] (array '[[T1]]) bool)
CHECK: val: memq : (λ '[[T1:.+]] (array '[[T1]]) bool)
CHECK: val: find_opt : (λ (λ '[[T1:.+]] bool) (array '[[T1]]) (option '[[T1]]))
CHECK: val: find_index : (λ (λ '[[T1:.+]] bool) (array '[[T1]]) (option int))
CHECK: val: find_map : (λ (λ '[[T1:.+]] (option '[[T2:.+]])) (array '[[T1]]) (option '[[T2]]))
CHECK: val: find_mapi : (λ (λ int '[[T1:.+]] (option '[[T2:.+]])) (array '[[T1]]) (option '[[T2]]))
CHECK: val: split : (λ (array (* '[[T1:.+]] '[[T2:.+]])) (* (array '[[T1]]) (array '[[T2]])))
CHECK: val: combine : (λ (array '[[T1:.+]]) (array '[[T2:.+]]) (array (* '[[T1]] '[[T2]])))
CHECK: val: sort : (λ (λ '[[T1:.+]] '[[T1]] int) (array '[[T1]]) unit)
CHECK: val: stable_sort : (λ (λ '[[T1:.+]] '[[T1]] int) (array '[[T1]]) unit)
CHECK: val: fast_sort : (λ (λ '[[T1:.+]] '[[T1]] int) (array '[[T1]]) unit)
*)

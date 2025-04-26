val map : ('a -> 'b) -> 'a list -> 'b list
val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b list -> 'a
val fold_right : ('a -> 'b -> 'b) -> 'a list -> 'b -> 'b
val length : 'a list -> int
val rev : 'a list -> 'a list
val append : 'a list -> 'a list -> 'a list
val concat : 'a list list -> 'a list
val iter : ('a -> unit) -> 'a list -> unit


type int
type float
type string
type unit
type bool
type 'a list
type 'a array
type 'a option = None | Some of 'a

val sqrt : float -> float
val print_int : int -> unit
val print_endline : string -> unit
val print_newline : unit -> unit
val print_string : string -> unit
val print_int : int -> unit
val print_float : float -> unit
val string_of_int : int -> string
val float_of_int : int -> float
val int_of_float : float -> int
val ( + ) : int -> int -> int
val ( - ) : int -> int -> int
val ( * ) : int -> int -> int
val ( / ) : int -> int -> int
val ( % ) : int -> int -> int
val ( < ) : int -> int -> bool
val ( *. ) : float -> float -> float
val ( /. ) : float -> float -> float
val ( <. ) : float -> float -> bool
val ( +. ) : float -> float -> float
val ( -. ) : float -> float -> float
val ( <= ) : int -> int -> bool
val ( > ) : int -> int -> bool
val ( >= ) : int -> int -> bool
val ( = ) : int -> int -> bool
val ( <> ) : int -> int -> bool
val ( && ) : bool -> bool -> bool
val ( || ) : bool -> bool -> bool
val ( @@ ) : ('a -> 'b) -> 'a -> 'b
val ( ^ ) : string -> string -> string
val ( = ) : 'a -> 'a -> bool
val ( <> ) : 'a -> 'a -> bool
val ( < ) : 'a -> 'a -> bool

module Hello = struct
  let message = "Hello from Florence"
  let print () = print_endline message
end

module M2 : sig
  val print : unit -> unit
end = struct
  let message = "Hello from M2"
  let print () = print_endline message
end


let () =
  Hello.print ();
  M2.print ()

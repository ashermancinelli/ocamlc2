module Hello = struct
  let message = "Hello from Florence"
  let print () = print_endline message
end

let () =
  Hello.print ()
  (*
  Hello.print ();
  print_goodbye ()
  *)

module Hello = struct
  let message = "Hello from Florence"
  let print () = print_endline message
  let p2 (x : int) = print_int (x + 1)
end

let print_goodbye () = print_endline "Goodbye"

let () =
  Hello.p2 1;
  Hello.print ();
  print_goodbye ()

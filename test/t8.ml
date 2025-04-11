let rec f x = if x > 5 then x else f (x + 1) in print_int (f 0);;

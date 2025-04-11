let rec f x = if x > 5 then x else f (x + 1);;
print_int (f 0);;

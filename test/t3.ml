let f lb ub = 
    let x = lb in
    let y = ub in
        for i = x to y do
            print_int i
        done;;

let x = 1 in f x 1;;

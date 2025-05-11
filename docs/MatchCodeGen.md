For generating a match expression, we create an scf.execute_region with a terminating block taking one ocaml.obox parameter and scf.yield-ing it.

We jump to the first block, and if we have a match, we enter an scf.if

```ocaml
match x with
  | Some y -> y
  | None -> 0
```

```mlir
func.func @match(%x: !ocaml.obox<i64>) -> !ocaml.unit {
    scf.execute_region {
        // create a dummy based on the match pattern
        %unit = ocaml.unit()
        %some_y = func.call @optional_some(%x)
        %pat1 = func.call @caml_match(%x, %some_y)
        cf.cond_br %pat1, ^some_bb(%some_y), ^none_bb(%none_y)
    ^some_bb(%some_y: !ocaml.obox):
        cf.br ^return_bb(%some_y)
    ^none_bb(%none_y: !ocaml.obox):
        %none_y = func.call @optional_none(%x)
        %pat2 = func.call @caml_match(%x, %none_y)
        cf.cond_br %pat2, ^return_bb(%none_y), ^assert_bb()
    ^assert_bb:
        %false = arith.constant false : i1
        cf.assert %false, "non-exhaustive match"
    ^return_bb(%ret: !ocaml.obox):
        scf.yield %ret
    }
}
```

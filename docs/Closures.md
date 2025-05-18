# Closure-lowering plan

This is the strategy I propose for representing OCaml closures in MLIR using the new `ocaml.env.*` operations.

1. **Environment creation (definition site)**  
   • Allocate a fresh `ocaml.env.new` immediately before the `func` that will become a closure.  
   • Keep the SSA handle around so we can populate it and pass it to the lifted function.
2. **Capture population**  
   • For each free variable `x` in the function body, emit `ocaml.env.capture %x in %env` in the definition's scope.  
   • Maintain a deterministic ordering so loads inside the body know the index (array-like semantics).
3. **Function representation**
   -   Two options:
   -   a. Re-write every captured function to take `%env : !ocaml.env` as an extra leading argument.  This keeps us compatible with existing `func`/`call` infrastructure.  
   -   b. Introduce an `ocaml.closure` op that models a function + implicit environment.  Simpler conceptually, but requires custom call lowering.  
   -   I lean toward (a) for now—minimal new IR surface.
   We will introduce a dedicated `ocaml.func` op that is explicitly closure-aware.

   • Signature still lists the *user-visible* parameters.  
   • The captured environment is accessed via an *implicit* SSA value `%env : !ocaml.env` that is added as the **first** block argument of the entry block and surfaced through a dedicated accessor on the op (`getEnv()` in C++).

   Rationale: keeps user signatures clean, avoids rewriting every call site with a hidden parameter, and gives optimisations a first-class handle to recognise closures.

   A sketch of the TableGen definition:

   ```td
   def Ocaml_FuncOp : Ocaml_Op<"func", [
       Symbol, FunctionLike, CallableOpInterface, RegionRecursiveSideEffects
     ]> {
     let summary = "Closure-aware function";
     let arguments = (ins);
     let results   = (outs);
     let regions   = (region SizedRegion<1>:$body);
     let hasCustomAssemblyFormat = 1; // we print like builtin func but omit %env
     let extraClassDeclaration = [{
       mlir::Value getEnv() {
         return getBody().front().getArgument(0);
       }
     }];
   }
   ```

   Example IR (pretty-printed):

   ```mlir
   ocaml.func @add (i64 %x, i64 %y) -> i64 {
     // implicit %0 : !ocaml.env is the first argument
     %lhs   = ocaml.env.get %0[0] -> i64     // captured value
     %sum   = arith.addi %lhs, %x : i64
     return %sum : i64
   }
   ```

   The *definition* site creates/initialises the env and then materialises a closure value:

   ```mlir
   %env  = ocaml.env.new
   %env1 = ocaml.env.capture %a in %env : i64
   %f    = ocaml.closure @add with %env1 : (i64, i64) -> i64 // helper op
   ```

   Calls become:

   ```mlir
   %result = ocaml.invoke %f(%arg1, %arg2) : (i64, i64) -> i64
   ```

   (`ocaml.closure`/`ocaml.invoke` are thin wrappers that unpack the pair and forward to the underlying `ocaml.func`.)

   This avoids contaminating existing `func.call` lowering and keeps closure semantics clear.
4. **Load inside the body**  
   • Replace uses of each free variable with `ocaml.env.get %env[idx] -> <type>` at the top of the function.  
   • Values then flow normally.  This preserves SSA dominance.
5. **Rewrite call sites**  
   • When the function value is taken (e.g., passed or stored), package the pair `<symref, env>` into an `ocaml.box` (or a dedicated closure type).  
   • Direct calls at the definition site can still be inlined without an env.

---

## Pitfalls / open questions

* **Environment lifetime:** A captured env must outlive all closures that reference it.  We currently allocate it in the defining block; GC / deallocation semantics are TBD.
* **Mutability:** Capturing a `ref` cell vs capturing the value makes a difference.  We need alias analysis to decide whether to load once or on every access.
* **Recursive & mutually-recursive functions:** These need the env before the function symbol is materialised—may require a two-phase build (declare then populate).
* **Nested closures:** A closure that itself defines an inner closure must thread its own env plus outer env(s).  We can either chain envs or flatten captures.
* **Type opacity:** `ocaml.env` is opaque; the optimizer must not assume any layout but still allow DCE when captures are unused.
* **ABI impact:** Adding a hidden param changes call sites.  We need a Conversion pass that rewrites `ocaml.call` sites consistently.
* **Partial application:** OCaml allows partial application which itself creates a closure.  Our machinery must be reusable here.

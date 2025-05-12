# Closure Captures in the OCaml Dialect

This document describes **phase 1** of closure support for the `ocaml` MLIR
backend.  Instead of threading extra parameters through every nested
function, we represent captured values with a *global lookup table* that is
written at the closure-creation site and read from inside the function body.

---

## 1. Dialect changes

### 1.1  `ocaml.global`
```
ocaml.global @<sym_name> : <type>
```
Module-level symbol that allocates storage for one captured value.
It has **no operands or results**; the stored type is provided as a
`TypeAttr`.  Multiple `global_load` / `global_store` operations can access it.

### 1.2  `ocaml.global_load`
```
%v = ocaml.global_load @<sym_name> : <type>
```
Pure operation that returns the value currently stored in the global.

### 1.3  `ocaml.global_store`
```
ocaml.global_store %new_val to @<sym_name> : <type>
```
Writes a new value into the global.  It has *memory write* side effects so
that the optimiser cannot move or eliminate it past reads.

See the updated TableGen definitions for the exact assembly formats and
traits.

### 1.4  `ocaml.global_load`
```
%v = ocaml.global_load @<sym_name> : <type>
```
Pure operation that returns the value currently stored in the global.

### 1.5  `ocaml.global_store`
```
ocaml.global_store %new_val to @<sym_name> : <type>
```
Writes a new value into the global.  It has *memory write* side effects so
that the optimiser cannot move or eliminate it past reads.

See the updated TableGen definitions for the exact assembly formats and
traits.

---

## 2. Lowering strategy in `MLIRGen3`

### 2.1  Terminology
* **Closure-creating expression** – the `(fun … -> …)` that produces a
  first-class function value.
* **Capture slot** – the `ocaml.global` op that stores one particular free
  variable.

### 2.2  Algorithm (per closure literal)
1.  **Collect free variables** of the lambda body *after* the type-inference
    pass.  (We already walk the subtree for MLIR generation, so we can reuse
    that helper.)
2.  For each free variable `v`:
    a.  Compute a *stable* symbol name, e.g. `@closure_⟨id⟩_v` where `id` is
        the AST node ID of the lambda.  Collisions are impossible inside one
        compilation unit.
    b.  **Create** an `ocaml.global` op in the module (if it does not already
        exist) with the MLIR type of `v`.
    c.  **Store** the current SSA value of `v` into the global *immediately
        before* the closure literal yields its result:
        ```cpp
        builder.create<ocaml::GlobalStoreOp>(loc, vValue, globalSym);
        ```
3.  Materialise the closure value.  For phase 1 we still expose the inner
    function as a `func.func` whose body *loads* its captures at the top:

    ```cpp
    auto loaded = builder.create<ocaml::GlobalLoadOp>(loc, globalSym, vType);
    declareVariable("v", loaded, loc);
    ```
4.  Calls to the closure (`func.call @g`) need **no extra operands**.  The
   function body itself fetches the captures when it runs.

### 2.3  Example
Input programme:
```ocaml
let f x y =
  let g = (fun z -> x + z) in
  let x = 5 in
  g y
```
Lowering excerpt (simplified):
```mlir
// Capture slot created once per closure literal
ocaml.global @closure_42_x : !ocaml.box<i64>

func.func @g(%z: !ocaml.box<i64>) -> !ocaml.box<i64> {
  %x = ocaml.global_load @closure_42_x : !ocaml.box<i64>
  %sum = func.call @+(%x, %z) : (!ocaml.box<i64>, !ocaml.box<i64>) -> !ocaml.box<i64>
  func.return %sum
}

func.func @f(%x0: !ocaml.box<i64>, %y: !ocaml.box<i64>) -> !ocaml.box<i64> {
  // store capture when g is *created*
  ocaml.global_store %x0 to @closure_42_x : !ocaml.box<i64>
  // shadowing of x afterwards does not affect the stored value
  %five = ocaml.convert 5 : i64 -> !ocaml.box<i64>
  // … code for "let x = 5 in" …
  %res = func.call @g(%y) : (!ocaml.box<i64>) -> !ocaml.box<i64>
  func.return %res
}
```

### 2.4  Properties
* **Exactly OCaml semantics** – the value is frozen at closure-creation; later
  shadowing / mutation (except through references) is invisible to the
  closure.
* **Single-threaded OK** – every call site overwrites the same slot, but the
  order is deterministic.
* **Multi-threading caveat** – sharing a single slot is a data race.  We will
  address this in phase 2 by switching to per-closure allocations using
  `ocaml.make_closure` once the dialect supports boxed closures.

### 2.5  Name-mangling & shadowing

Each closure literal receives a **unique numeric id** (`lambdaId`) at MLIR
generation time.  Capture slots are named

```
closure_<lambdaId>__<var-name>
```

• `lambdaId` is a monotonically increasing counter or the AST node id of the
  `(fun … -> …)` expression.  Two different lambdas can never collide even if
  they live under the same parent function.

• The human-readable `var-name` makes IR dumps easier to inspect but plays no
  role in uniqueness.

Why this works with shadowing:

```
let f x =            (* lambdaId = 1 *)
  let g y =          (* lambdaId = 2 *)
    let h z =        (* lambdaId = 3 *)
      x + y + z      (* captures x and y from outer scopes *)
    in h 0
  in
  let x = 5 in       (* shadows the parameter *)
  g 7
```

* `g` stores its capture under `closure_2__x`.
* `h` stores two separate captures `closure_3__x`, `closure_3__y`.

All three slots differ because the prefix differs.  The fact that the inner
`let x = 5` shadows the parameter happens **after** the store executed for
`g`, hence does not affect it.  No run-time stack walk is needed: the compiler
knows the correct binding statically.

---

## 3. Future work
1.  Replace the ad-hoc global table with first-class *closure* values that own
    their own environment (more faithful, thread-safe, GC-friendly).
2.  Add a verifier to check that load/store types match the declared global.
3.  Optimise away unused capture slots (DCE pass).

## 4.  Contrast with `fir.global` in Flang

Flang's `fir.global` (see `FIROps.td` and its lowering in `CodeGen.cpp`) serves a **very different purpose** from the lightweight `ocaml.global` we introduce for closure captures.

| Aspect | `ocaml.global` (this design) | `fir.global` (Flang) |
|--------|------------------------------|----------------------|
|Primary role | Scratch slot that **holds the run-time value of a captured variable**.| Define **static program data** (variables, constants, type-info blobs) that exist for the whole program/image.|
|Lifetime | Created once per closure literal **but written at run-time every time a closure is built**.  Value is mutable and expected to change between calls. | One per symbol in the compile unit.  Usually constant‐initialised and seldom mutated at run-time (unless user wrote a global variable).|
|Initialization | No regional initializer; the slot is **uninitialised** until the first `global_store`. | Rich initializer support: constant attribute or an explicit region with FIR ops and terminator `fir.has_value`.|
|IR size | Zero operands/results, only `sym_name` + `typeAttr`.  Very small. | Carries full type, optional initial value, linkage attrs, data attributes, alignment, etc.  Potentially large region.|
|Memory effects | Accessed via dedicated `global_load` / `global_store` ops that model read/write side-effects. | Lowered directly to `llvm.global`; reads/writes use plain `llvm.load`/`llvm.store` on its address; effect modelling inherited from those ops.|
|Lowering path | Stays in OCaml dialect until capture-rewriting pass; later becomes an `llvm.global` or an alloca-like object depending on final scheme.  For now we expect a trivial lowering pass that turns it into `llvm.global` with `internal` linkage and no initializer. | Already has a full lowering implementation (`GlobalOpConversion` in `CodeGen.cpp`) that handles constants vs variables, COMDAT, address-spaces, CUDA shared memory, etc.|
|Thread-safety | **Not thread-safe**: every store overwrites the single slot.  Acceptable in current single-threaded prototype; documented caveat. | Same as regular globals – thread-safe if program uses appropriate synchronisation.|
|Design philosophy | Temporary mechanism to avoid free-variable analysis at call sites; will eventually be replaced by proper closure environment objects. | Final representation for Fortran globals – part of the language semantics, not a stepping stone.|

### Implications

* We purposely keep `ocaml.global` minimal: no constant folding, no data-layout attributes, no linkage keywords.  Flang needs all that because the objects it models must survive to the final object file unchanged.
* The load/store pair makes capture semantics explicit at the SSA level, whereas FIR relies on standard `llvm.load/store` once the address of the global is materialised.
* `fir.global` can host *dynamic* initialisation code in its region (mutable globals start as zeros, constants hold their value); our capture slots cannot – they are populated from normal IR just after they are created.

In short, `ocaml.global` is a **run-time capture slot**, not a general global-data facility.  The overlap in the keyword "global" is superficial; operational semantics, lifetime and lowering path differ markedly from Flang's `fir.global`.

### 5.  Re-assessment after the Flang comparison

Seeing how heavyweight and semantically "final" `fir.global` is reinforces that our tiny `ocaml.global` is **only suitable as a stop-gap**.  The comparison surfaces a few limitations we should plan to address:

1.  **Re-entrancy / multi-threading**  
    A single mutable slot cannot safely serve two closures that are alive at the same time (recursive call, threads, async).  We either need:
    • a per-closure allocation op (e.g. `ocaml.make_closure`) that returns a pointer to a freshly malloc'ed environment, or  
    • to revert to the extra-parameter approach once we have free-variable analysis.

2.  **Optimisation barriers**  
    Because the slot is memory, not SSA, passes like LICM or GVN cannot see through the loads/stores.  A future environment-object representation could still be SSA (env pointer is SSA, fields accessed via `llvm.load` but alias scope is narrower).

3.  **Captured value lifetime**  
    The global stays alive for the whole module even after the closure dies; with many literals this becomes bloat.  Per-instance storage would match OCaml semantics better and make it easier for the GC once we have one.

4.  **Verification**  
    We should add a verifier that  
    • ensures each `global_store` precedes the first `global_load`, and  
    • checks type consistency.

5.  **Naming & linkage**  
    Using module-level symbols (`@closure_42_x`) is OK for now but will clash after inlining or LTO.  A dedicated attribute like `private, internal, linkonce_odr` would be safer, or hide the storage inside the forthcoming closure object.

6.  **Migration path**  
    Retain `ocaml.global` for quick progress but keep the API surface minimal so we can automate conversion to the eventual closure dialect.

**Short-term tweaks**

* Mark the op `IsolatedFromAbove` so that loads/stores cannot reach outside the module during canonicalisation.
* Add an optional `thread_local` attribute to sidestep races if the runtime becomes multi-threaded before we have proper environments.

**Long-term direction**

Move toward a first-class closure construct:

```
ocaml.make_closure @g_impl(%x_capture) : (!ocaml.box<i64>) -> !ocaml.closure<( !ocaml.box<i64> ) -> !ocaml.box<i64>>
```

This would allocate an environment object and return an SSA value that carries both the code pointer and the captures, allowing inlining, DCE and better alias analysis.

Until then the current global-table scheme remains acceptable for single-threaded experiments, but we should treat it as **technical debt**.

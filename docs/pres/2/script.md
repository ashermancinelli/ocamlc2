# OCaml Type Inference & "If You See What I Mean" (IYSWIM)

## 1. Cold-open: the problem

```plaintext
"Python lets me write x + y without thinking; C++ makes me spell out every type.  Can I have the best of both worlds—*if-you-see-what-I-mean*?"
```

Show side-by-side:

```python
# Python (dynamic)
def add(x, y):
    return x + y
```

```ocaml
(* OCaml *)
let add x y = x + y;;
```

Then reveal compiler / interpreter responses:

```bash
$ python -c "import add; print(add.add(1,2))"   # runs (dynamic)
$ ocamlc -c add.ml                                # compiles, infers: val add : int -> int -> int
```

Hook: OCaml keeps **brevity** of Python while retaining **safety** of C++.

Two extremes so far:

• **Python** – a name can hold *any* value; the interpreter only complains *at run-time*.

```python
x = 3
x = "hello"        # still fine
print(x + 1)        # boom – TypeError only when executed
```

• **C** – every variable *must* be declared and the compiler does almost no guessing.

```c
int add(int x, int y) { return x + y; }
void* v = (void*)add;      // escape hatch: throw away type info
```

for flexibility we can throw away type info with `void*`.

**If-you-see-what-I-mean**, we want something in the middle:

* Write concise code like Python.
* Keep compile-time guarantees like C (without `void*` foot-guns).
* Let the compiler *solve a Sudoku* to deduce every missing type.

That middle path is exactly what OCaml's Hindley–Milner inference provides.

---

## 2. "IYSWIM" Moment – what we actually want

* Safety at compile time (no surprises in prod)
* No annotation overhead (just write the code, the compiler solves Sudoku for you)
* Performance: statically known representations

"So, if-you-see-what-I-mean, we want **safety _and_ brevity**."

---

## 3. Walk-through example (`let x f y = f y`)

Use Manim caret animation:

1. Introduce fresh vars `'a`, `'b`, `'c` …
2. Add constraints (`f` must be a function; `y` matches its arg type …).
3. Unify → principal type:

```ocaml
val x : ('c -> 'd) -> 'c -> 'd
```

Inline comment: Hindley–Milner ≈ type Sudoku.

---

## 4. Lift the curtain: Hindley-Milner in one slide

* Algorithm W (Cardelli / Milner)
* Environment Γ, judgement `Γ ⊢ e : τ`
* Unification step (`doUnify` in `Unifier.cpp`)

_"Don't panic: we'll stay high-level."_

---

## 4b. Under the hood – Type Variables & Unification

"Sudoku" is really three small C++ helpers in our compiler:

1. **TypeOperator** – a concrete constructor like `int`, `list`, or `→` that can *contain* other types
2. **TypeVariable** – an *unknown* we will later solve for
3. **prune** – follow variable links until you hit a root
4. **unify** – recursively make two trees identical, setting pointers along the way

```cpp
// TypeSystem.h
struct TypeOperator : public TypeExpr {
  std::string_view name;            // e.g. "int", "list", "function"
  std::vector<TypeExpr*> args;      // sub-types (may include variables)
};

struct TypeVariable : public TypeExpr {
  TypeExpr *instance = nullptr;  // becomes non-null when solved
};

// Unifier.cpp (simplified)
TypeExpr* prune(TypeExpr* t) {
  if (auto *tv = dyn_cast<TypeVariable>(t); tv && tv->instance)
      return tv->instance = prune(tv->instance);
  return t;
}

LogicalResult Unifier::unify(TypeExpr* a, TypeExpr* b){
  a = prune(a); b = prune(b);
  if(auto *tva = dyn_cast<TypeVariable>(a)) {
      if (a != b) tva->instance = b;          // instantiate variable
  } else if(auto *toa = dyn_cast<TypeOperator>(a)) {
      // pair-wise unify operator args …
  }
}
```

Cloning (for polymorphism) just makes fresh `TypeVariable`s before entering another scope:

```cpp
// Inference.cpp
TypeExpr* Unifier::clone(TypeExpr* t){
  if(auto *tv = dyn_cast<TypeVariable>(t) && isGeneric(tv))
      return createTypeVariable();  // α-renaming
  // recurse on TypeOperator children …
}
```

• **Type variables** give us the blanks.  
• **Prune** keeps the trees short.  
• **Unify** fills the blanks consistently.  
• **Clone** lets a polymorphic value get fresh blanks each time it's used.

Armed with those four helpers, the whole HM engine in `Inference.cpp` reduces to:

1. Walk the syntax tree, create vars & operators.
2. Add constraints (`unify`).
3. At every use site of a bound identifier, `clone` to avoid accidental sharing.

That's the machinery behind the animation you just saw.

---

## 5. Compare languages

| Language      | Need to write types? | Strict? | How it manages |
|---------------|---------------------|---------|----------------|
| Python        | Never               | No      | Tags at runtime |
| TypeScript    | Optional            | Erased  | Gradual        |
| Rust          | Sometimes           | Yes     | Local inference + lifetimes |
| **OCaml**     | **Rarely**          | **Yes** | Hindley–Milner |

Key takeaway: **principal types** mean the compiler finds the most general type automatically.

---

## 6. When inference *fails*

```ocaml
let tricky x y = x y;;
```

Compiler error shows *which* usage is ambiguous. Safety preserved.

---

## 7. Beyond HM (teasers)

* System F: make polymorphism explicit.
* Row polymorphism → records & variants.
* Functors = modules as functions (demo slides).
* GADTs, effects in OCaml 5.

---

## 9. Call-to-action

Links:

* Luca Cardelli – "Basic Polymorphic Typechecking"
* Peter Landin – "The Next 700 Programming Languages"
* OCaml manual on polymorphic variants & modules
* MLIR Toy example (why C++ infrastructure matters)

"OCaml feels dynamically typed *if you squint*, but the moment you squint wrong the compiler saves you — **IYSWIM**."

---

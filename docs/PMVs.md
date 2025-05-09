# Implementing Polymorphic Variants and Row Polymorphism

This document outlines a detailed plan for extending ocamlc2's type system to support OCaml's polymorphic variants and row polymorphism.

## 1. Introduction to Polymorphic Variants

Unlike regular variants in OCaml, polymorphic variants do not require pre-declaration of variant types. They're prefixed with a backtick (`` ` ``) and can be used flexibly across different contexts. For example:

```ocaml
(* Using polymorphic variants without declaration *)
let x = `A
let y = `B 5
let process = function
  | `A -> "Got A"
  | `B n -> "Got B: " ^ string_of_int n
```

The key features that make polymorphic variants powerful:
- Tags don't belong to any predefined type
- Types are inferred from usage
- Open/extensible variant types
- Support for subtyping relationships

## 2. Row Polymorphism Theory

Row polymorphism is the type-theoretic foundation that enables polymorphic variants. It introduces:

1. **Row variables**: Type variables representing "the rest of the fields/tags"
2. **Presence/absence information**: Tracking which tags must be present or may be absent
3. **Extensibility markers**: Indicating whether a type can be extended ([>], [<], or [])

The OCaml type notation reflects this:
- `[> `A | `B]` means "at least tags `A` and `B`" (extensible)
- `[< `A | `B]` means "at most tags `A` and `B`" (closed)
- `[`A | `B]` means "exactly tags `A` and `B`" (fixed)

## 3. Implementation Plan for ocamlc2

### 3.1 New TypeExpr Subclasses

```cpp
// New class for polymorphic variants
class PolymorphicVariantOperator : public TypeOperator {
public:
  enum Kind {
    Fixed,    // []
    AtLeast,  // [>]
    AtMost    // [<]
  };

private:
  Kind kind;
  TypeVariable *rowVar;  // Row variable for extensibility
  std::map<llvm::StringRef, TypeExpr*> tags;  // Tag -> Type mapping

public:
  // Constructor
  PolymorphicVariantOperator(Kind k, 
                             llvm::ArrayRef<std::pair<llvm::StringRef, TypeExpr*>> tags,
                             TypeVariable *rowVar = nullptr);
                             
  // Type operator interface                          
  void print(llvm::raw_ostream &os) const override;
  bool equals(const TypeExpr *other) const override;
  
  // PMV-specific methods
  Kind getKind() const { return kind; }
  bool isExtensible() const { return kind == AtLeast; }
  bool isClosed() const { return kind == AtMost; }
  bool isFixed() const { return kind == Fixed; }
  
  ArrayRef<std::pair<llvm::StringRef, TypeExpr*>> getTags() const;
  TypeVariable *getRowVar() const { return rowVar; }
  
  // Row operations
  void addTag(llvm::StringRef tag, TypeExpr *type);
  PolymorphicVariantOperator *merge(PolymorphicVariantOperator *other);
  PolymorphicVariantOperator *intersect(PolymorphicVariantOperator *other);
};

// For intersection types (e.g., `A of int & string)
class IntersectionType : public TypeExpr {
private:
  SmallVector<TypeExpr*> types;
  
public:
  IntersectionType(ArrayRef<TypeExpr*> types);
  ArrayRef<TypeExpr*> getTypes() const { return types; }
  
  void print(llvm::raw_ostream &os) const override;
  bool equals(const TypeExpr *other) const override;
};
```

### 3.2 Unification for Polymorphic Variants

```cpp
LogicalResult Unifier::unifyPolymorphicVariants(
    PolymorphicVariantOperator *a, 
    PolymorphicVariantOperator *b) {
  
  // Case 1: [> `A | `B] and [> `B | `C] => [> `A | `B | `C]
  if (a->isExtensible() && b->isExtensible()) {
    // Union of tags (preserve constraints from both sides)
    auto merged = a->merge(b);
    
    // Unify row variables to propagate constraints
    if (a->getRowVar() && b->getRowVar()) {
      ORFAIL(unify(a->getRowVar(), b->getRowVar()));
    }
    
    // Replace a and b with the merged variant in any substitutions
    // ...
    
    return success();
  }
  
  // Case 2: [< `A | `B] and [< `B | `C] => [< `B]
  else if (a->isClosed() && b->isClosed()) {
    // Intersection of tags (only common tags are allowed)
    auto intersected = a->intersect(b);
    
    // Cannot have empty intersection if both are used in the same context
    if (intersected->getTags().empty()) {
      return failure();
    }
    
    // Replace a and b with the intersected variant in any substitutions
    // ...
    
    return success();
  }
  
  // Case 3: [> `A | `B] and [< `B | `C] => [`B]
  else if (a->isExtensible() && b->isClosed()) {
    // Check that all tags in a are in b
    for (auto &[tag, type] : a->getTags()) {
      if (!b->hasTag(tag)) {
        return failure();
      }
    }
    
    // Result is fixed with intersection of tags
    auto fixed = createFixed(a, b);
    
    // Replace a and b with the fixed variant
    // ...
    
    return success();
  }
  
  // Case 4: [< `A | `B] and [> `B | `C] => [`B]
  else {
    // Same as case 3 but swapped
    return unifyPolymorphicVariants(b, a);
  }
}

// Also handle unifying tag parameter types 
// (e.g., `A of int unifying with `A of int & string)
LogicalResult Unifier::unifyVariantTags(
    llvm::StringRef tag, TypeExpr *typeA, TypeExpr *typeB) {
  
  if (!typeA)
    return success();  // No parameter to unify
    
  if (!typeB)
    return failure();  // Parameter vs. no parameter
    
  // If either is an intersection type, handle specially
  if (auto *intersectA = llvm::dyn_cast<IntersectionType>(typeA)) {
    // ...
  }
  
  // Otherwise, regular unification
  return unify(typeA, typeB);
}
```

### 3.3 Parser and AST Integration

Update the AST parser to recognize:

1. **Polymorphic variant constructors**: `` `Tag`` and `` `Tag(expr)``
2. **Polymorphic variant types**: `[< `A | `B]` and `[> `A | `B]`
3. **Intersection types**: `'a & 'b`

```cpp
TypeExpr* Unifier::inferPolymorphicVariantConstructor(Cursor ast) {
  auto node = ast.getCurrentNode();
  std::string tag = getTextWithoutBacktick(node.getNamedChild(0));
  
  TypeExpr *paramType = nullptr;
  if (node.getNumNamedChildren() > 1) {
    paramType = infer(node.getNamedChild(1));
  }
  
  // Create an "at least this tag" type, extensible with a fresh row variable
  auto rowVar = createTypeVariable();
  return create<PolymorphicVariantOperator>(
      PolymorphicVariantOperator::AtLeast,
      {{tag, paramType}},
      rowVar);
}

TypeExpr* Unifier::inferPolymorphicVariantType(Cursor ast) {
  auto node = ast.getCurrentNode();
  bool isAtMost = node.getNamedChild(0).getType() == "<";
  auto kind = isAtMost ? 
      PolymorphicVariantOperator::AtMost : 
      PolymorphicVariantOperator::AtLeast;
  
  // Parse variant tags and their types
  std::map<std::string, TypeExpr*> tags;
  for (auto tagNode : node.getNamedChildren()) {
    if (tagNode.getType() == "polymorphic_variant_tag") {
      std::string tag = getTextWithoutBacktick(tagNode.getNamedChild(0));
      TypeExpr *paramType = nullptr;
      if (tagNode.getNumNamedChildren() > 1) {
        paramType = infer(tagNode.getNamedChild(1));
      }
      tags[tag] = paramType;
    }
  }
  
  // Create the polymorphic variant type
  auto rowVar = createTypeVariable();
  return create<PolymorphicVariantOperator>(kind, tags, rowVar);
}
```

### 3.4 Enhancing Pattern Matching

Update the pattern matching implementation to handle polymorphic variants:

```cpp
TypeExpr* Unifier::inferMatchCase(TypeExpr* matcheeType, ts::Node node) {
  // Existing code...
  
  // Special handling for polymorphic variant patterns
  if (pattern.getType() == "polymorphic_variant_pattern") {
    // Create a closed variant type [< `Tag1 | `Tag2 | ... ]
    auto patternType = inferPolymorphicVariantPattern(pattern);
    
    // Unify with matchee type, which constrains what the matchee can be
    UNIFY_OR_RNULL(matcheeType, patternType);
    
    // Continue with standard match case handling...
  }
  
  // Existing code...
}
```

### 3.5 Coercions and Or-patterns

Support for explicit type coercions like `(expr :> type)`:

```cpp
TypeExpr* Unifier::inferCoercion(Cursor ast) {
  auto node = ast.getCurrentNode();
  auto expr = node.getNamedChild(0);
  auto type = node.getNamedChild(1);
  
  // Infer the type of the expression
  auto *exprType = infer(expr);
  
  // Parse the target type
  auto *targetType = inferType(type);
  
  // Check subtyping relationship (specialized for polymorphic variants)
  if (!isSubtype(exprType, targetType)) {
    // Error: invalid coercion, types not compatible
    return nullptr;
  }
  
  return targetType;
}

// Subtyping for polymorphic variants
bool Unifier::isSubtype(TypeExpr *sub, TypeExpr *super) {
  if (auto *pvSub = llvm::dyn_cast<PolymorphicVariantOperator>(sub)) {
    if (auto *pvSuper = llvm::dyn_cast<PolymorphicVariantOperator>(super)) {
      // [> `A | `B] is a subtype of [> `A]
      if (pvSuper->isExtensible()) {
        // All tags in super must be in sub
        // ...
      }
      
      // [< `A] is a subtype of [< `A | `B]
      if (pvSub->isClosed()) {
        // All tags in sub must be in super
        // ...
      }
      
      // Other cases...
    }
  }
  
  // Default case
  return false;
}
```

## 4. Generalization and Instantiation

Row variables need special handling during generalization and instantiation:

```cpp
TypeExpr* Unifier::generalize(TypeExpr* type, Env& env) {
  // Existing generalization code...
  
  // Special handling for polymorphic variants
  if (auto *pv = llvm::dyn_cast<PolymorphicVariantOperator>(type)) {
    // Generalize row variables not bound in the environment
    if (auto *rowVar = pv->getRowVar()) {
      if (isFreeInEnv(rowVar, env)) {
        // Quantify over the row variable when creating the type scheme
        // ...
      }
    }
    
    // Generalize parameter types
    for (auto &[tag, paramType] : pv->getTags()) {
      if (paramType) {
        generalize(paramType, env);
      }
    }
  }
  
  // Continue with standard generalization...
}
```

## 5. Implementation Challenges

### 5.1 Subtyping and Variance

Polymorphic variants introduce subtyping into the type system, which complicates unification:

- `[> `A | `B]` is a subtype of `[> `A]` (more tags → more specific)
- `[< `A]` is a subtype of `[< `A | `B]` (fewer tags → more specific)

The unifier must handle these relationships correctly.

### 5.2 Intersection Types

When multiple patterns constrain the same tag with different types:

```ocaml
let f = function
  | `A x when x > 0 -> x  (* x: int *)
  | `A s -> String.length s  (* s: string *)
```

This should create a type `[< `A of int & string | ... ]` for the argument.

### 5.3 Or-patterns with Aliases

The OCaml manual describes special behavior for or-patterns with aliases:

```ocaml
match x with
| (`A | `B) as tag -> ...
```

The alias `tag` only gets the tags in the or-pattern (`A` and `B`), not the whole type of `x`.

### 5.4 Row Variables and Unification

Row variables have special unification rules:
- They can be instantiated to include more tags
- They must maintain constraints from their uses

## 6. Testing Plan

Implement test cases covering:

1. **Basic polymorphic variants**:
   ```ocaml
   let x = `A;;
   let y = `B 5;;
   ```

2. **Pattern matching**:
   ```ocaml
   let f = function
     | `A -> "A"
     | `B n -> "B" ^ string_of_int n
   ```

3. **Subtyping relationships**:
   ```ocaml
   let f (x : [< `A | `B]) = match x with `A -> () | `B -> ();;
   f (`A :> [> `A | `C])  (* Should work *)
   ```

4. **Intersection types**:
   ```ocaml
   let f = function
     | `A x when x > 0 -> x
     | `A s -> String.length s
   ```

5. **Or-patterns**:
   ```ocaml
   let f = function
     | (`A | `B) as tag -> ...
   ```

## 7. Future Extensions

Once basic polymorphic variants are implemented, we can consider:

1. **Object types**: Another application of row polymorphism
2. **First-class polymorphism**: For more flexible variant handling
3. **Type-based dispatch**: Functions dispatched based on variant types
4. **Type-level programming**: Using variants at the type level

## References

- [OCaml Manual - Polymorphic variants](https://ocaml.org/manual/5.3/polyvariant.html)
- [Efficient and Insightful Generalization](https://okmij.org/ftp/ML/generalization.html)
- [Row Polymorphism Section 10.8](http://gallium.inria.fr/~fpottier/publis/emlti-final.pdf)

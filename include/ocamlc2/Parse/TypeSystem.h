#pragma once

#include "ocamlc2/Parse/AST.h"
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <set>
#include <llvm/Support/Casting.h>
#include <llvm/ADT/ScopedHashTable.h>
namespace ocamlc2 {

struct TypeExpr {
  enum Kind {
    Operator,
    Variable,
  };
  TypeExpr(Kind kind) : kind(kind) {}
  virtual ~TypeExpr() = default;
  virtual llvm::StringRef getName() const = 0;
  Kind getKind() const { return kind; }
  bool operator==(const TypeExpr& other) const;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type);
private:
  Kind kind;
};

struct TypeOperator : public TypeExpr {
  TypeOperator(llvm::StringRef name, llvm::ArrayRef<TypeExpr*> args={}) : TypeExpr(Kind::Operator), args(args), name(name) {}
  inline llvm::StringRef getName() const override { return name; }
  inline llvm::ArrayRef<TypeExpr*> getArgs() const { return args; }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Operator; }
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op);
  inline TypeExpr* at(size_t index) const { return args[index]; }
  consteval static llvm::StringRef getFunctionOperatorName() { return "->"; }
  consteval static llvm::StringRef getTupleOperatorName() { return "*"; }

private:
  llvm::SmallVector<TypeExpr*> args;
  std::string name;
};

struct FunctionOperator : public TypeOperator {
  FunctionOperator(TypeExpr* from, TypeExpr* to) : TypeOperator(TypeOperator::getFunctionOperatorName(), {from, to}) {}
};

struct TupleOperator : public TypeOperator {
  TupleOperator(llvm::ArrayRef<TypeExpr*> args) : TypeOperator(TypeOperator::getTupleOperatorName(), args) {}
};

struct TypeVariable : public TypeExpr {
  TypeVariable();
  inline llvm::StringRef getName() const override { 
    if (not name) {
      name = std::string("T" + std::to_string(id));
    }
    return *name;
  }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Variable; }
  inline bool instantiated() const { return instance != nullptr; }
  bool operator==(const TypeVariable& other) const;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeVariable& var);
  TypeExpr* instance = nullptr;
private:
  int id;
  mutable std::optional<std::string> name = std::nullopt;
};

struct Function : TypeOperator {
  Function(TypeExpr* from, TypeExpr* to)
    : TypeOperator("->", {from, to}) {}
};

struct Unifier {
  using Env = llvm::ScopedHashTable<llvm::StringRef, TypeExpr*>;
  template<typename T>
  using Set = llvm::DenseSet<T>;
  Env env;
  void initializeEnvironment();
  TypeExpr* infer(const ASTNode* ast);
  template <typename T, typename... Args>
  T* create(Args&&... args) {
    static_assert(std::is_base_of_v<TypeExpr, T>,
                  "Unifier should only be used to create TypeExpr subclasses");
    typeArena.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    return static_cast<T*>(typeArena.back().get());
  }
private:
  void unify(TypeExpr* a, TypeExpr* b);

  // Clone a type expression, replacing generic type variables with new ones
  TypeExpr* clone(TypeExpr* type);

  // The function Prune is used whenever a type expression has to be inspected: it will always
  // return a type expression which is either an uninstantiated type variable or a type operator; i.e. it
  // will skip instantiated variables, and will actually prune them from expressions to remove long
  // chains of instantiated variables.
  TypeExpr* prune(TypeExpr* type);

  TypeExpr* inferType(const ASTNode* ast);

  inline bool isSubType(TypeExpr* a, TypeExpr* b) {
    b = prune(b);
    if (auto *op = llvm::dyn_cast<TypeOperator>(b)) {
      return isSubTypeOfAny(a, op->getArgs());
    } else if (llvm::isa<TypeVariable>(b)) {
      return *a == *b;
    }
    assert(false && "Unknown type expression");
  }

  template<typename ITERABLE>
  inline bool isSubTypeOfAny(TypeExpr* type, ITERABLE types) {
    return llvm::any_of(types, [this, type](TypeExpr* other) {
      return isSubType(type, other);
    });
  }

  template<typename ITERABLE>
  inline bool isGeneric(TypeExpr* type, ITERABLE concreteTypes) {
    return !isSubTypeOfAny(type, concreteTypes);
  }

  template<typename ITERABLE>
  inline bool isConcrete(TypeExpr* type, ITERABLE concreteTypes) {
    return isSubTypeOfAny(type, concreteTypes);
  }

  inline TypeExpr* getDeclaredType(const llvm::StringRef name) {
    if (env.count(name)) {
      return clone(env.lookup(name));
    }
    assert(false && "Type not declared");
    return nullptr;
  }

  inline auto *createFunction(TypeExpr* from, TypeExpr* to) {
    return create<FunctionOperator>(from, to);
  }

  inline auto *createTuple(llvm::ArrayRef<TypeExpr*> args) {
    return create<TupleOperator>(args);
  }

  Set<TypeVariable*> concreteTypes;
  std::vector<std::unique_ptr<TypeExpr>> typeArena;
};

}

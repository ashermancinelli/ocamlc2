#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <set>
#include <llvm/Support/Casting.h>
#include <ocamlc2/Parse/ScopedHashTable.h>

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
  template<typename T> friend T& operator<<(T& os, const TypeExpr& type);
private:
  Kind kind;
};

struct TypeOperator : public TypeExpr {
  TypeOperator(llvm::StringRef name, llvm::ArrayRef<TypeExpr*> args={}) : TypeExpr(Kind::Operator), args(args), name(name) {}
  inline llvm::StringRef getName() const override { return name; }
  inline llvm::ArrayRef<TypeExpr*> getArgs() const { return args; }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Operator; }
  template<typename T> friend T& operator<<(T& os, const TypeOperator& op);
  inline TypeExpr* at(size_t index) const { return args[index]; }
  consteval static llvm::StringRef getFunctionOperatorName() { return "λ"; }
  consteval static llvm::StringRef getTupleOperatorName() { return "*"; }
  consteval static llvm::StringRef getConstructorOperatorName() { return "V"; }
  consteval static llvm::StringRef getListOperatorName() { return "List"; }
  consteval static llvm::StringRef getArrayOperatorName() { return "Array"; }
  consteval static llvm::StringRef getRecordOperatorName() { return "Record"; }
  inline TypeExpr *back() const { return args.back(); }

private:
  llvm::SmallVector<TypeExpr*> args;
  std::string name;
};

struct FunctionOperator : public TypeOperator {
  FunctionOperator(llvm::ArrayRef<TypeExpr*> args) : TypeOperator(TypeOperator::getFunctionOperatorName(), args) {}
};

struct TupleOperator : public TypeOperator {
  TupleOperator(llvm::ArrayRef<TypeExpr*> args) : TypeOperator(TypeOperator::getTupleOperatorName(), args) {}
  static inline bool classof(const TypeExpr *expr) {
    if (expr->getKind() == Kind::Operator) {
      auto *op = llvm::cast<TypeOperator>(expr);
      return op->getName() == TypeOperator::getTupleOperatorName();
    }
    return false;
  }
};

struct TypeVariable : public TypeExpr {
  TypeVariable();
  inline llvm::StringRef getName() const override { 
    if (not name) {
      name = std::string("'t" + std::to_string(id));
    }
    return *name;
  }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Variable; }
  inline bool instantiated() const { return instance != nullptr; }
  bool operator==(const TypeVariable& other) const;
  template<typename T> friend T& operator<<(T& os, const TypeVariable& var);
  TypeExpr* instance = nullptr;
private:
  int id;
  mutable std::optional<std::string> name = std::nullopt;
};

template<typename T>
T& operator<<(T& os, const TypeOperator& op) {
  auto args = op.getArgs();
  auto name = op.getName().str();
  if (auto pos = name.find("StdlibMM"); pos != std::string::npos) {
    name = name.substr(pos + 8);
  }
  if (args.empty()) {
    return os << name;
  }
  os << '(' << op.getName();
  for (auto *arg : args) {
    os << ' ' << *arg;
  }
  return os << ')';
}

template<typename T>
T& operator<<(T& os, const TypeExpr& type) {
  if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    os << *to;
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(&type)) {
    os << *tv;
  }
  return os;
}

template<typename T>
T& operator<<(T& os, const TypeVariable& var) {
  if (var.instantiated()) {
    os << *var.instance;
  } else {
    os << var.getName();
  }
  return os;
}

}

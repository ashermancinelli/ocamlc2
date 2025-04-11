#pragma once

#include "ocamlc2/Parse/AST.h"
#include <memory>
#include <string>
#include <unordered_map>
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
  virtual std::string getName() const = 0;
  Kind getKind() const { return kind; }
private:
  Kind kind;
};

struct TypeOperator : public TypeExpr {
  TypeOperator(std::string name, std::vector<TypeExpr*> args) : TypeExpr(Kind::Operator), args(args), name(name) {}
  inline std::string getName() const override { return name; }
  inline std::vector<TypeExpr*> getArgs() const { return args; }
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Operator; }
private:
  std::vector<TypeExpr*> args;
  std::string name;
};

struct TypeVariable : public TypeExpr {
  TypeVariable();
  inline std::string getName() const override { return "T" + std::to_string(id); }
  std::optional<TypeExpr*> instance = std::nullopt;
  static inline bool classof(const TypeExpr* expr) { return expr->getKind() == Kind::Variable; }
private:
  int id;
};

struct Function : TypeOperator {
  Function(TypeExpr* from, TypeExpr* to)
    : TypeOperator("->", {from, to}) {}
};

std::ostream& operator<<(std::ostream& os, const TypeExpr& type);
std::ostream& operator<<(std::ostream& os, const TypeOperator& op);
std::ostream& operator<<(std::ostream& os, const TypeVariable& var);

struct Unifier {
  using Env = llvm::ScopedHashTable<llvm::StringRef, std::unique_ptr<TypeExpr>>;
  using EnvScope = Env::ScopeTy;
  std::set<TypeExpr*> concreteTypes;
  Env env;
  void unify(TypeExpr& a, TypeExpr& b);
  TypeExpr* infer(const ASTNode* ast){return nullptr;}
};

}

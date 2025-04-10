#pragma once

#include "ocamlc2/Parse/AST.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <set>
#include <llvm/Support/Casting.h>



namespace ocamlc2 {

// Forward declarations
class Type;
class TypeVar;
class TypeCon;
class TypeApp;
class TypeArrow;
class TypeScheme;
class TypeEnv;
class TypeConstraint;
class TypeInferenceContext;

// Type class - base class for all types
class Type {
public:
  enum TypeKind {
    Kind_Var,    // Type variable (e.g., 'a, 'b)
    Kind_Con,    // Type constructor (e.g., int, bool)
    Kind_App,    // Type application (e.g., list int)
    Kind_Arrow,  // Function type (e.g., int -> bool)
  };

  Type(TypeKind kind) : kind(kind) {}
  virtual ~Type() = default;

  TypeKind getKind() const { return kind; }
  virtual std::string toString() const = 0;
  virtual std::set<std::string> getFreeTypeVars() const = 0;
  virtual std::shared_ptr<Type> substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const = 0;

private:
  TypeKind kind;
};

// Type variable (e.g., 'a, 'b)
class TypeVar : public Type {
public:
  TypeVar(std::string name) : Type(Kind_Var), name(std::move(name)) {}

  const std::string& getName() const { return name; }
  std::string toString() const override { return name; }
  std::set<std::string> getFreeTypeVars() const override { return {name}; }
  std::shared_ptr<Type> substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const override;

  static bool classof(const Type* type) { return type->getKind() == Kind_Var; }

private:
  std::string name;
};

// Type constructor (e.g., int, bool)
class TypeCon : public Type {
public:
  TypeCon(std::string name) : Type(Kind_Con), name(std::move(name)) {}

  const std::string& getName() const { return name; }
  std::string toString() const override { return name; }
  std::set<std::string> getFreeTypeVars() const override { return {}; }
  std::shared_ptr<Type> substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const override;

  static bool classof(const Type* type) { return type->getKind() == Kind_Con; }

private:
  std::string name;
};

// Type application (e.g., list int)
class TypeApp : public Type {
public:
  TypeApp(std::shared_ptr<Type> constructor, std::shared_ptr<Type> argument)
    : Type(Kind_App), constructor(std::move(constructor)), argument(std::move(argument)) {}

  const std::shared_ptr<Type>& getConstructor() const { return constructor; }
  const std::shared_ptr<Type>& getArgument() const { return argument; }
  std::string toString() const override;
  std::set<std::string> getFreeTypeVars() const override;
  std::shared_ptr<Type> substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const override;

  static bool classof(const Type* type) { return type->getKind() == Kind_App; }

private:
  std::shared_ptr<Type> constructor;
  std::shared_ptr<Type> argument;
};

// Function type (e.g., int -> bool)
class TypeArrow : public Type {
public:
  TypeArrow(std::shared_ptr<Type> from, std::shared_ptr<Type> to)
    : Type(Kind_Arrow), from(std::move(from)), to(std::move(to)) {}

  const std::shared_ptr<Type>& getFrom() const { return from; }
  const std::shared_ptr<Type>& getTo() const { return to; }
  std::string toString() const override;
  std::set<std::string> getFreeTypeVars() const override;
  std::shared_ptr<Type> substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const override;

  static bool classof(const Type* type) { return type->getKind() == Kind_Arrow; }

private:
  std::shared_ptr<Type> from;
  std::shared_ptr<Type> to;
};

// Type scheme (polymorphic type, e.g., forall a. a -> a)
class TypeScheme {
public:
  TypeScheme(std::vector<std::string> vars, std::shared_ptr<Type> type)
    : vars(std::move(vars)), type(std::move(type)) {}

  const std::vector<std::string>& getVars() const { return vars; }
  const std::shared_ptr<Type>& getType() const { return type; }
  std::string toString() const;
  std::set<std::string> getFreeTypeVars() const;
  std::shared_ptr<Type> instantiate(TypeInferenceContext& context) const;

private:
  std::vector<std::string> vars;
  std::shared_ptr<Type> type;
};

// Type environment (mapping from variable names to type schemes)
class TypeEnv {
public:
  TypeEnv() = default;

  void extend(const std::string& name, std::shared_ptr<TypeScheme> scheme) {
    env[name] = std::move(scheme);
  }

  std::shared_ptr<TypeScheme> lookup(const std::string& name) const {
    auto it = env.find(name);
    if (it != env.end()) {
      return it->second;
    }
    return nullptr;
  }

  std::set<std::string> getFreeTypeVars() const;
  TypeEnv substitute(const std::unordered_map<std::string, std::shared_ptr<Type>>& subst) const;

private:
  std::unordered_map<std::string, std::shared_ptr<TypeScheme>> env;
};

// Type constraint (equality constraint between two types)
class TypeConstraint {
public:
  TypeConstraint(std::shared_ptr<Type> lhs, std::shared_ptr<Type> rhs)
    : lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  const std::shared_ptr<Type>& getLHS() const { return lhs; }
  const std::shared_ptr<Type>& getRHS() const { return rhs; }
  std::string toString() const;

private:
  std::shared_ptr<Type> lhs;
  std::shared_ptr<Type> rhs;
};

// Type inference context
class TypeInferenceContext {
public:
  TypeInferenceContext() : nextVarId(0) {}

  std::shared_ptr<Type> freshTypeVar() {
    std::string name = "'t" + std::to_string(nextVarId++);
    return std::make_shared<TypeVar>(name);
  }

  std::shared_ptr<Type> inferType(const ASTNode* node, TypeEnv& env);
  std::shared_ptr<TypeScheme> generalize(const TypeEnv& env, const std::shared_ptr<Type>& type);
  std::unordered_map<std::string, std::shared_ptr<Type>> unify(const std::vector<TypeConstraint>& constraints);

private:
  int nextVarId;
  std::vector<TypeConstraint> constraints;

  // Helper functions for type inference
  std::shared_ptr<Type> inferExpr(const ASTNode* node, TypeEnv& env);
  std::shared_ptr<Type> inferValuePath(const ValuePathAST* node, TypeEnv& env);
  std::shared_ptr<Type> inferNumberExpr(const NumberExprAST* node, TypeEnv& env);
  std::shared_ptr<Type> inferApplication(const ApplicationExprAST* node, TypeEnv& env);
  std::shared_ptr<Type> inferInfixExpr(const InfixExpressionAST* node, TypeEnv& env);
  std::shared_ptr<Type> inferMatchExpr(const MatchExpressionAST* node, TypeEnv& env);
  std::shared_ptr<Type> inferLetExpr(const LetExpressionAST* node, TypeEnv& env);
  std::shared_ptr<Type> inferLetBinding(const LetBindingAST* node, TypeEnv& env);
  
  // Occurs check for unification
  bool occurs(const std::string& var, const std::shared_ptr<Type>& type);
  
  // Unification algorithm
  std::unordered_map<std::string, std::shared_ptr<Type>> unifyOne(
      const std::shared_ptr<Type>& t1, 
      const std::shared_ptr<Type>& t2,
      std::unordered_map<std::string, std::shared_ptr<Type>> subst);
};

// Utility functions
std::shared_ptr<TypeScheme> inferProgramType(const ASTNode* ast);
void dumpType(const std::shared_ptr<Type>& type);
void dumpTypeScheme(const std::shared_ptr<TypeScheme>& scheme);

} // namespace ocamlc2 

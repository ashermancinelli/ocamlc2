#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Support/LLVMCommon.h"
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
#include <ocamlc2/Parse/TypeSystem.h>
#include <cpp-tree-sitter.h>

namespace ocamlc2 {
inline namespace ts {
using namespace ::ts;
namespace detail {
  struct ModuleSearchPathScope;
  struct ModuleScope;
  struct Scope;
}
struct Unifier {
  Unifier(std::string_view source) : source(source) {}
  Unifier() {}
  llvm::raw_ostream &show(ts::Cursor cursor, bool showUnnamed = false);
  using Env = llvm::ScopedHashTable<llvm::StringRef, TypeExpr *>;
  using EnvScope = Env::ScopeTy;
  using ConcreteTypes = llvm::DenseSet<TypeVariable *>;
  TypeExpr *infer(Cursor ast);
  TypeExpr *infer(ts::Node const &ast);
  template <typename T, typename... Args> T *create(Args &&...args) {
    static_assert(std::is_base_of_v<TypeExpr, T>,
                  "Unifier should only be used to create TypeExpr subclasses");
    typeArena.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    return static_cast<T *>(typeArena.back().get());
  }

private:
  void initializeEnvironment();

  LogicalResult unify(TypeExpr *a, TypeExpr *b);

  // Clone a type expression, replacing generic type variables with new ones
  TypeExpr *clone(TypeExpr *type);

  // The function Prune is used whenever a type expression has to be inspected:
  // it will always return a type expression which is either an uninstantiated
  // type variable or a type operator; i.e. it will skip instantiated variables,
  // and will actually prune them from expressions to remove long chains of
  // instantiated variables.
  TypeExpr *prune(TypeExpr *type);

  TypeExpr *inferType(Cursor ast);
  TypeExpr *inferValuePath(Cursor ast);
  TypeExpr *inferConstructorPath(Cursor ast);
  TypeExpr *inferLetBinding(Cursor ast);
  TypeExpr *inferLetExpression(Cursor ast);
  TypeExpr *inferForExpression(Cursor ast);
  TypeExpr *inferCompilationUnit(Cursor ast);
  TypeExpr *inferApplicationExpression(Cursor ast);
  TypeExpr *inferInfixExpression(Cursor ast);
  TypeExpr *inferIfExpression(Cursor ast);
  TypeExpr *inferMatchExpression(Cursor ast);
  TypeExpr *inferMatchCase(TypeExpr *matcheeType, ts::Node node);
  TypeExpr *inferPattern(ts::Node node);
  TypeExpr *inferGuard(Cursor ast);
  TypeExpr *inferArrayGetExpression(Cursor ast);
  TypeExpr *inferListExpression(Cursor ast);

  bool isSubType(TypeExpr *a, TypeExpr *b);

  template <typename ITERABLE>
  inline bool isSubTypeOfAny(TypeExpr *type, ITERABLE types) {
    return llvm::any_of(types, [this, type](TypeExpr *other) {
      return isSubType(type, other);
    });
  }

  template <typename ITERABLE>
  inline bool isGeneric(TypeExpr *type, ITERABLE concreteTypes) {
    return !isSubTypeOfAny(type, concreteTypes);
  }

  template <typename ITERABLE>
  inline bool isConcrete(TypeExpr *type, ITERABLE concreteTypes) {
    return isSubTypeOfAny(type, concreteTypes);
  }

  TypeExpr *declare(Node node, TypeExpr *type);
  TypeExpr *declare(llvm::StringRef name, TypeExpr *type);
  TypeExpr *declarePath(llvm::ArrayRef<llvm::StringRef> path, TypeExpr *type);
  TypeExpr *declarePatternVariables(const ASTNode *ast,
                                    llvm::SmallVector<TypeExpr *> &typevars);
  inline bool declared(llvm::StringRef name) { return env.count(name) > 0; }

  TypeExpr *getType(Node node);
  TypeExpr *getType(const llvm::StringRef name);
  TypeExpr *getType(std::vector<std::string> path);
  llvm::SmallVector<TypeExpr *> getParameterTypes(Cursor parameters);

  inline auto *createFunction(llvm::ArrayRef<TypeExpr *> args) {
    return create<FunctionOperator>(args);
  }

  inline auto *createTuple(llvm::ArrayRef<TypeExpr *> args) {
    return create<TupleOperator>(args);
  }

  inline auto *createTypeVariable() { return create<TypeVariable>(); }

  inline auto *createTypeOperator(llvm::StringRef name,
                                  llvm::ArrayRef<TypeExpr *> args = {}) {
    return create<TypeOperator>(name, args);
  }

  inline auto *getBoolType() { return getType("bool"); }
  inline auto *getFloatType() { return getType("float"); }
  inline auto *getIntType() { return getType("int"); }
  inline auto *getUnitType() { return getType("unit!"); }
  inline auto *getStringType() { return getType("string"); }
  inline auto *getWildcardType() { return getType("_"); }
  inline auto *getVarargsType() { return getType("varargs!"); }
  inline auto *getListTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getListOperatorName(), {type});
  }
  inline auto *getListType() { return getListTypeOf(createTypeVariable()); }
  inline auto *getArrayTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getArrayOperatorName(), {type});
  }
  inline auto *getArrayType() { return getArrayTypeOf(createTypeVariable()); }
  bool isVarargs(TypeExpr *type);
  bool isWildcard(TypeExpr *type);

  void pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> modules);
  inline void pushModuleSearchPath(llvm::StringRef module) {
    pushModuleSearchPath(llvm::ArrayRef{module});
  }
  void pushModule(llvm::StringRef module);
  void popModuleSearchPath();
  void popModule();
  std::string getHashedPath(llvm::ArrayRef<llvm::StringRef> path);
  std::vector<std::string> getPathParts(Node node);

  std::string_view source;
  Env env;
  ConcreteTypes concreteTypes;
  std::vector<std::unique_ptr<TypeExpr>> typeArena;
  llvm::SmallVector<llvm::StringRef> moduleSearchPath;
  llvm::SmallVector<llvm::StringRef> currentModule;
  llvm::DenseMap<ts::NodeID, TypeExpr *> nodeToType;
  friend struct detail::ModuleSearchPathScope;
  friend struct detail::ModuleScope;
  friend struct detail::Scope;
};

namespace detail {
struct ModuleSearchPathScope {
  ModuleSearchPathScope(Unifier &unifier,
                        llvm::ArrayRef<llvm::StringRef> modules)
      : unifier(unifier) {
    unifier.pushModuleSearchPath(modules);
  }
  ~ModuleSearchPathScope() { unifier.popModuleSearchPath(); }

private:
  Unifier &unifier;
};
struct ModuleScope {
  ModuleScope(Unifier &unifier, llvm::StringRef module) : unifier(unifier) {
    unifier.pushModule(module);
    unifier.pushModuleSearchPath({module});
  }
  ~ModuleScope() {
    unifier.popModule();
    unifier.popModuleSearchPath();
  }

private:
  Unifier &unifier;
};

struct Scope {
  Scope(Unifier *unifier)
      : unifier(unifier), envScope(unifier->env),
        concreteTypes(unifier->concreteTypes) {}
  ~Scope() { unifier->concreteTypes = std::move(concreteTypes); }

private:
  Unifier *unifier;
  Unifier::EnvScope envScope;
  Unifier::ConcreteTypes concreteTypes;
};

} // namespace detail
} // namespace ts
} // namespace ocamlc2

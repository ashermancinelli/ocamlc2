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
#include <ocamlc2/Parse/TypeSystem.h>

namespace ocamlc2 {
inline namespace old {
struct Unifier {
  Unifier() { }
  using Env = llvm::ScopedHashTable<llvm::StringRef, TypeExpr*>;
  using EnvScope = Env::ScopeTy;
  using ConcreteTypes = llvm::DenseSet<TypeVariable*>;
  TypeExpr* infer(const ASTNode* ast);
  template <typename T, typename... Args>
  T* create(Args&&... args) {
    static_assert(std::is_base_of_v<TypeExpr, T>,
                  "Unifier should only be used to create TypeExpr subclasses");
    typeArena.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    return static_cast<T*>(typeArena.back().get());
  }
private:
  void initializeEnvironment();

  void unify(TypeExpr* a, TypeExpr* b);

  // Clone a type expression, replacing generic type variables with new ones
  TypeExpr* clone(TypeExpr* type);

  // The function Prune is used whenever a type expression has to be inspected: it will always
  // return a type expression which is either an uninstantiated type variable or a type operator; i.e. it
  // will skip instantiated variables, and will actually prune them from expressions to remove long
  // chains of instantiated variables.
  TypeExpr* prune(TypeExpr* type);

  TypeExpr* inferType(const ASTNode* ast);

  bool isSubType(TypeExpr* a, TypeExpr* b);

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

  TypeExpr* declare(llvm::StringRef name, TypeExpr* type);
  TypeExpr* declarePath(llvm::ArrayRef<llvm::StringRef> path, TypeExpr* type);
  TypeExpr* declarePatternVariables(const ASTNode* ast, llvm::SmallVector<TypeExpr*>& typevars);
  inline bool declared(llvm::StringRef name) {
    return env.count(name) > 0;
  }

  TypeExpr* getType(const llvm::StringRef name);
  TypeExpr* getType(std::vector<std::string> path);
  llvm::SmallVector<TypeExpr *>
  getParameterTypes(const std::vector<std::unique_ptr<ASTNode>> &parameters);

  inline auto *createFunction(llvm::ArrayRef<TypeExpr*> args) {
    return create<FunctionOperator>(args);
  }

  inline auto *createTuple(llvm::ArrayRef<TypeExpr*> args) {
    return create<TupleOperator>(args);
  }

  inline auto *createTypeVariable() {
    return create<TypeVariable>();
  }

  inline auto *createTypeOperator(llvm::StringRef name, llvm::ArrayRef<TypeExpr*> args={}) {
    return create<TypeOperator>(name, args);
  }

  inline auto *getBoolType() { return getType("bool"); }
  inline auto *getFloatType() { return getType("float"); }
  inline auto *getIntType() { return getType("int"); }
  inline auto *getUnitType() { return getType("unit!"); }
  inline auto *getStringType() { return getType("string"); }
  inline auto *getWildcardType() { return getType("_"); }
  inline auto *getVarargsType() { return getType("varargs!"); }
  inline auto *getListType() { return getType("List"); }
  inline auto *getListOf(TypeExpr* type) { return createTypeOperator("List", {type}); }
  bool isVarargs(TypeExpr* type);
  bool isWildcard(TypeExpr* type);

  void pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> modules);
  inline void pushModuleSearchPath(llvm::StringRef module) {
    pushModuleSearchPath(llvm::ArrayRef{module});
  }
  void pushModule(llvm::StringRef module);
  void popModuleSearchPath();
  void popModule();
  std::string getHashedPath(llvm::ArrayRef<llvm::StringRef> path);

  struct ModuleSearchPathScope {
    ModuleSearchPathScope(Unifier& unifier, llvm::ArrayRef<llvm::StringRef> modules) : unifier(unifier) {
      unifier.pushModuleSearchPath(modules);
    }
    ~ModuleSearchPathScope() {
      unifier.popModuleSearchPath();
    }
  private:
    Unifier& unifier;
  };
  struct ModuleScope {
    ModuleScope(Unifier& unifier, llvm::StringRef module) : unifier(unifier) {
      unifier.pushModule(module);
      unifier.pushModuleSearchPath({module});
    }
    ~ModuleScope() {
      unifier.popModule();
      unifier.popModuleSearchPath();
    }
  private:
    Unifier& unifier;
  };

  Env env;
  ConcreteTypes concreteTypes;
  std::vector<std::unique_ptr<TypeExpr>> typeArena;
  llvm::SmallVector<llvm::StringRef> moduleSearchPath;
  llvm::SmallVector<llvm::StringRef> currentModule;
};

struct TypeCheckingPass : public ASTPass {
  void run(CompilationUnitAST *node) override {
    unifier.infer(node);
  }
private:
  Unifier unifier;
};

}
} // namespace ocamlc2

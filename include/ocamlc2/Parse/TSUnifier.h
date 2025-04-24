#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Parse/TSUtil.h"
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
#include <filesystem>
#include <llvm/Support/Casting.h>
#include <ocamlc2/Parse/ScopedHashTable.h>
#include <ocamlc2/Parse/TypeSystem.h>
#include <cpp-tree-sitter.h>

namespace ocamlc2 {
inline namespace ts {
using namespace ::ts;
namespace fs = std::filesystem;
namespace detail {
  struct ModuleSearchPathScope;
  struct ModuleScope;
  struct Scope;
}
struct Unifier {
  Unifier();
  Unifier(std::string filepath);
  void loadSourceFile(fs::path filepath);
  void loadImplementationFile(fs::path filepath);
  void loadInterfaceFile(fs::path filepath);
  void loadStdlibInterfaces(fs::path exe);
  void dumpTypes(llvm::raw_ostream &os);
  llvm::raw_ostream &show(ts::Cursor cursor, bool showUnnamed = false);
  using Env = llvm::ScopedHashTable<llvm::StringRef, TypeExpr *>;
  using EnvScope = Env::ScopeTy;
  using ConcreteTypes = llvm::DenseSet<TypeVariable *>;
  TypeExpr *infer(Cursor ast);
  TypeExpr *infer(ts::Node const &ast);
  inline auto *getInferredType(ts::Node const &ast) {
    return nodeToType.lookup(ast.getID());
  }
  template <typename T, typename... Args> T *create(Args &&...args) {
    static_assert(std::is_base_of_v<TypeExpr, T>,
                  "Unifier should only be used to create TypeExpr subclasses");
    typeArena.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    return static_cast<T *>(typeArena.back().get());
  }
  TypeExpr *getType(Node node);
  TypeExpr *getType(ts::NodeID id);
  inline std::string_view getText(Node node) {
    return ts::getText(node, sources.back().source);
  }
  llvm::StringRef getTextSaved(Node node);

private:
  inline auto *createTypeVariable() { return create<TypeVariable>(); }

  inline auto *createTypeOperator(llvm::StringRef name,
                                  llvm::ArrayRef<TypeExpr *> args = {}) {
    return create<TypeOperator>(name, args);
  }

public:

  TypeExpr *getBoolType();
  TypeExpr *getFloatType();
  TypeExpr *getIntType();
  TypeExpr *getUnitType();
  TypeExpr *getStringType();
  TypeExpr *getWildcardType();
  TypeExpr *getVarargsType();
  inline auto *getFunctionType(llvm::ArrayRef<TypeExpr *> args) {
    return create<FunctionOperator>(args);
  }
  FunctionOperator *getFunctionTypeForPartialApplication(FunctionOperator *func, unsigned arity);
  inline auto *getTupleType(llvm::ArrayRef<TypeExpr *> args) {
    return createTypeOperator(TypeOperator::getTupleOperatorName(), args);
  }
  inline auto *getListTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getListOperatorName(), {type});
  }
  inline auto *getListType() { return getListTypeOf(createTypeVariable()); }
  inline auto *getArrayTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getArrayOperatorName(), {type});
  }
  inline auto *getArrayType() { return getArrayTypeOf(createTypeVariable()); }
  inline auto *getRecordType(ArrayRef<TypeExpr *> fields) {
    return createTypeOperator(TypeOperator::getRecordOperatorName(), fields);
  }
  bool isVarargs(TypeExpr *type);
  bool isWildcard(TypeExpr *type);

  struct SourceFile {
    fs::path filepath;
    std::string source;
    ::ts::Tree tree;
  };
  SmallVector<SourceFile> sources;

private:
  void initializeEnvironment();

  LogicalResult unify(TypeExpr *a, TypeExpr *b);

  // Clone a type expression, replacing generic type variables with new ones
  TypeExpr *clone(TypeExpr *type);
  TypeExpr *clone(TypeExpr *type, llvm::DenseMap<TypeExpr *, TypeExpr *> &mapping);

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
  TypeExpr *inferLetBindingFunction(Node name, SmallVector<Node> parameters, Node body);
  TypeExpr *inferLetBindingRecursiveFunction(Node name, SmallVector<Node> parameters, Node body);
  TypeExpr *inferLetBindingValue(Node name, Node body);
  TypeExpr *inferLetExpression(Cursor ast);
  TypeExpr *inferForExpression(Cursor ast);
  TypeExpr *inferCompilationUnit(Cursor ast);
  TypeExpr *inferApplicationExpression(Cursor ast);
  TypeExpr *inferConcatExpression(Cursor ast);
  TypeExpr *inferInfixExpression(Cursor ast);
  TypeExpr *inferIfExpression(Cursor ast);
  TypeExpr *inferMatchExpression(Cursor ast);
  TypeExpr *inferMatchCase(TypeExpr *matcheeType, ts::Node node);
  TypeExpr *inferPattern(ts::Node node);
  TypeExpr *inferTuplePattern(ts::Node node);
  TypeExpr *inferProductExpression(Cursor ast);
  TypeExpr *inferValuePattern(Cursor ast);
  TypeExpr *inferParenthesizedPattern(Cursor ast);
  TypeExpr *inferConstructorPattern(Cursor ast);
  TypeExpr *inferGuard(Cursor ast);
  TypeExpr *inferArrayExpression(Cursor ast);
  TypeExpr *inferArrayGetExpression(Cursor ast);
  TypeExpr *inferListExpression(Cursor ast);
  TypeExpr *inferFunctionExpression(Cursor ast);
  TypeExpr *inferModuleDefinition(Cursor ast);
  TypeExpr *inferModuleBinding(Cursor ast);
  TypeExpr *inferModuleSignature(Cursor ast);
  TypeExpr *inferModuleStructure(Cursor ast);
  TypeExpr *inferValueSpecification(Cursor ast);
  TypeExpr *inferFunctionSpecification(Cursor ast);
  TypeExpr *inferTypeExpression(Cursor ast);
  TypeExpr *inferTypeConstructorPath(Cursor ast);
  TypeExpr *inferSequenceExpression(Cursor ast);
  TypeExpr *inferTypeDefinition(Cursor ast);
  TypeExpr *inferTypeBinding(Cursor ast);
  TypeExpr *inferVariantConstructor(TypeExpr *variantType, Cursor ast);
  TypeExpr *inferRecordDeclaration(Cursor ast);
  TypeExpr *inferConstructedType(Cursor ast);
  TypeExpr *inferFunctionType(Cursor ast);
  TypeExpr *inferOpenModule(Cursor ast);
  TypeExpr *declareFunctionParameter(Node node);
  SmallVector<Node> flattenFunctionType(Node node);

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
  TypeExpr *declareConcrete(Node node);
  TypeExpr *declare(llvm::StringRef name, TypeExpr *type);
  TypeExpr *declarePath(llvm::ArrayRef<llvm::StringRef> path, TypeExpr *type);
  TypeExpr *declarePatternVariables(const ASTNode *ast,
                                    llvm::SmallVector<TypeExpr *> &typevars);
  inline bool declared(llvm::StringRef name) { return env.count(name); }

  TypeExpr *getType(const llvm::StringRef name);
  TypeExpr *maybeGetType(const llvm::StringRef name);
  TypeExpr *getType(std::string_view name);
  TypeExpr *getType(std::vector<std::string> path);
  TypeExpr *getType(const char *name);
  TypeExpr *setType(Node node, TypeExpr *type);

  // Does not error on missing typevariable because TVs are introduced implicitly
  // in value specifications and we need to create them on-demand, different from
  // function parameters. e.g. the following declaration does not declare the TV
  // before using it.
  //
  // val access : 'a list -> 'a
  TypeExpr *getTypeVariable(const llvm::StringRef name);
  TypeExpr *getTypeVariable(Node node);

  llvm::SmallVector<TypeExpr *> getParameterTypes(Cursor parameters);

  void pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> modules);
  inline void pushModuleSearchPath(llvm::StringRef module) {
    pushModuleSearchPath(llvm::ArrayRef{module});
  }
  void pushModule(llvm::StringRef module);
  void popModuleSearchPath();
  void popModule();
  std::string getHashedPath(llvm::ArrayRef<llvm::StringRef> path);
  std::vector<std::string> getPathParts(Node node);
  void maybeDumpTypes(Node node, TypeExpr *type);

  Env env;
  ConcreteTypes concreteTypes;
  std::vector<std::unique_ptr<TypeExpr>> typeArena;
  llvm::SmallVector<llvm::StringRef> moduleSearchPath;
  llvm::SmallVector<llvm::StringRef> currentModule;
  llvm::SmallVector<std::tuple<std::string, unsigned, ts::Node>> nodesToDump;
  llvm::DenseMap<ts::NodeID, TypeExpr *> nodeToType;
  llvm::DenseMap<StringRef, SmallVector<StringRef>> recordTypeFieldOrder;
  std::unique_ptr<detail::Scope> rootScope;
  bool isLoadingStdlib = false;
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

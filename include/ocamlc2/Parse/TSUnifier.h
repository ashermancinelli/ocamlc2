#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/Utils.h"
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
namespace fs = std::filesystem;
using ts::Node;
using ts::NodeID;
using ts::Cursor;
namespace detail {
  struct ModuleSearchPathScope;
  struct ModuleScope;
  struct Scope;
  struct TypeVariableScope;
}

struct Unifier {
  Unifier();

  // Construct a new unifier, load the standard library interfaces, load the
  // source file.
  Unifier(std::string filepath);

  llvm::raw_ostream &showType(llvm::raw_ostream &os, llvm::StringRef name);

  // Set the maximum number of errors that can be reported.
  // -1 indicates no limit.
  void setMaxErrors(int maxErrors);

  // Load a source file. The extension will be checked to determine if it is
  // an interface or implementation.
  void loadSourceFile(fs::path filepath);

  // Load a string directly as if it were a source file.
  void loadSource(llvm::StringRef source);

  // Load an implementation (.ml) file.
  void loadImplementationFile(fs::path filepath);

  // Load an interface (.mli) file.
  void loadInterfaceFile(fs::path filepath);

  // Load the standard library interfaces.
  void loadStdlibInterfaces(fs::path exe);

  // Show recorded function definitions, declarations, and variable let bindings,
  // for debugging and testing.
  void dumpTypes(llvm::raw_ostream &os);

  [[nodiscard]] bool anyFatalErrors() const;
  void showErrors();

  [[nodiscard]] operator LogicalResult() const {
    return failure(anyFatalErrors());
  }

  llvm::raw_ostream &showParseTree();
  llvm::raw_ostream &showTypedTree();
  llvm::raw_ostream &show(ts::Cursor cursor, bool showUnnamed = false, bool showTypes = true);
  llvm::raw_ostream &show(bool showUnnamed = false, bool showTypes = true);
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
  TypeExpr *getVariableType(ts::Node node);
  TypeExpr *getVariableType(ts::NodeID id);
  inline std::string_view getText(ts::Node node) {
    return ocamlc2::getText(node, sources.back().source);
  }
  llvm::StringRef getTextSaved(ts::Node node);

private:
  inline auto *createTypeVariable() { return create<TypeVariable>(); }

  inline auto *createTypeOperator(llvm::StringRef name,
                                  llvm::ArrayRef<TypeExpr *> args = {}) {
    return create<TypeOperator>(name, args);
  }
  inline auto *createTypeOperator(TypeOperator::Kind kind,
                                  llvm::StringRef name,
                                  llvm::ArrayRef<TypeExpr *> args = {}) {
    return create<TypeOperator>(kind, name, args);
  }

public:
  TypeExpr *getBoolType();
  TypeExpr *getFloatType();
  TypeExpr *getIntType();
  TypeExpr *getUnitType();
  TypeExpr *getStringType();
  TypeExpr *getWildcardType();
  TypeExpr *getVarargsType();
  inline auto *getFunctionType(
      llvm::ArrayRef<TypeExpr *> args,
      llvm::ArrayRef<ParameterDescriptor> parameterDescriptors = {}) {
    return create<FunctionOperator>(args, parameterDescriptors);
  }
  FunctionOperator *getFunctionTypeForPartialApplication(FunctionOperator *func, unsigned arity);

  std::pair<FunctionOperator *, TypeExpr *> normalizeFunctionType(TypeExpr *declaredType, SmallVector<Node> arguments);
  std::pair<SmallVector<std::pair<llvm::StringRef, Node>>, std::set<Node>> getLabeledArguments(SmallVector<Node> arguments);

  inline auto *getTupleType(llvm::ArrayRef<TypeExpr *> args) {
    return create<TupleOperator>(args);
  }
  inline auto *getListTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getListOperatorName(), {type});
  }
  inline auto *getListType() { return getListTypeOf(createTypeVariable()); }
  inline auto *getArrayTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getArrayOperatorName(), {type});
  }
  inline auto *getArrayType() { return getArrayTypeOf(createTypeVariable()); }
  inline auto *getOptionalTypeOf(TypeExpr *type) {
    return createTypeOperator(TypeOperator::getOptionalOperatorName(), {type});
  }
  inline auto *getOptionalType() {
    return getOptionalTypeOf(createTypeVariable());
  }
  inline auto *getRecordType(llvm::StringRef recordName, ArrayRef<TypeExpr *> fields, ArrayRef<llvm::StringRef> fieldNames) {
    return create<RecordOperator>(recordName, fields, fieldNames);
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
  LogicalResult initializeEnvironment();

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

  // \c infer is just a wrapper around this method which does some printing and
  // caches the type for a given node id. This method does the actual work.
  TypeExpr *inferType(Cursor ast);

  TypeExpr *inferApplicationExpression(Cursor ast);
  TypeExpr *inferArrayExpression(Cursor ast);
  TypeExpr *inferArrayGetExpression(Cursor ast);
  TypeExpr *inferCompilationUnit(Cursor ast);
  TypeExpr *inferConcatExpression(Cursor ast);
  TypeExpr *inferConstructedType(Cursor ast);
  TypeExpr *inferConstructorPath(Cursor ast);
  TypeExpr *inferConstructorPattern(Cursor ast);
  TypeExpr *inferExternal(Cursor ast);
  TypeExpr *inferForExpression(Cursor ast);
  TypeExpr *inferFunctionExpression(Cursor ast);
  TypeExpr *inferFunctionSpecification(Cursor ast);
  TypeExpr *inferFunctionType(Cursor ast);
  TypeExpr *inferGuard(Cursor ast);
  TypeExpr *inferIfExpression(Cursor ast);
  TypeExpr *inferInfixExpression(Cursor ast);
  TypeExpr *inferLabeledArgumentType(Cursor ast);
  TypeExpr *inferLetBinding(Cursor ast);
  TypeExpr *inferLetBindingFunction(Node name, SmallVector<Node> parameters, Node body);
  TypeExpr *inferLetBindingRecursiveFunction(Node name, SmallVector<Node> parameters, Node body);
  TypeExpr *inferLetBindingValue(Node name, Node body);
  TypeExpr *inferLetExpression(Cursor ast);
  TypeExpr *inferListExpression(Cursor ast);
  TypeExpr *inferMatchCase(TypeExpr *matcheeType, ts::Node node);
  TypeExpr *inferMatchExpression(Cursor ast);
  TypeExpr *inferModuleBinding(Cursor ast);
  TypeExpr *inferModuleDefinition(Cursor ast);
  TypeExpr *inferModuleSignature(Cursor ast);
  TypeExpr *inferModuleStructure(Cursor ast);
  TypeExpr *inferOpenModule(Cursor ast);
  TypeExpr *inferParenthesizedPattern(Cursor ast);
  TypeExpr *inferPattern(ts::Node node);
  TypeExpr *inferProductExpression(Cursor ast);
  TypeExpr *inferRecordDeclaration(llvm::StringRef recordName, Cursor ast);
  TypeExpr *inferRecordDeclaration(Cursor ast);
  TypeExpr *inferRecordExpression(Cursor ast);
  TypeExpr *inferSequenceExpression(Cursor ast);
  TypeExpr *inferTupleExpression(Cursor ast);
  TypeExpr *inferTuplePattern(ts::Node node);
  TypeExpr *inferTupleType(Cursor ast);
  TypeExpr *inferTypeBinding(Cursor ast);
  TypeExpr *inferTypeConstructorPath(Cursor ast);
  TypeExpr *inferTypeDefinition(Cursor ast);
  TypeExpr *inferTypeExpression(Cursor ast);
  TypeExpr *inferValuePath(Cursor ast);
  TypeExpr *inferValuePattern(Cursor ast);
  TypeExpr *inferValueSpecification(Cursor ast);
  TypeExpr *inferVariantConstructor(TypeExpr *variantType, Cursor ast);
  TypeExpr *inferVariantDeclaration(TypeExpr *variantType, Cursor ast);
  TypeExpr *inferFieldGetExpression(Cursor ast);

  TypeExpr *declareFunctionParameter(Node node);
  TypeExpr *declareFunctionParameter(ParameterDescriptor desc, Node node);
  ParameterDescriptor describeParameter(Node node);
  SmallVector<ParameterDescriptor> describeParameters(SmallVector<Node> parameters);
  FailureOr<ParameterDescriptor> describeFunctionArgumentType(Node node);

  inline SmallVector<Node> flattenFunctionType(Node node) {
    return flattenType("function_type", node);
  }
  inline SmallVector<Node> flattenTupleType(Node node) {
    return flattenType("tuple_type", node);
  }
  SmallVector<Node> flattenType(std::string_view nodeType, Node node);
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

  TypeExpr *declareVariable(Node node, TypeExpr *type);
  TypeExpr *declareConcreteVariable(Node node);
  TypeExpr *declareVariable(llvm::StringRef name, TypeExpr *type);
  TypeExpr *declareType(llvm::StringRef name, TypeExpr *type);
  TypeExpr *declareVariablePath(llvm::ArrayRef<llvm::StringRef> path, TypeExpr *type);
  TypeExpr *declarePatternVariables(const ASTNode *ast,
                                    llvm::SmallVector<TypeExpr *> &typevars);
  inline bool declared(llvm::StringRef name) { return env.count(name); }

  llvm::StringRef saveString(llvm::StringRef str);
  TypeExpr *getVariableType(const llvm::StringRef name);
  TypeExpr *maybeGetVariableType(const llvm::StringRef name);
  TypeExpr *maybeGetVariableTypeWithName(const llvm::StringRef name);
  TypeExpr *getVariableType(std::string_view name);
  TypeExpr *getVariableType(std::vector<std::string> path);
  TypeExpr *getVariableType(const char *name);
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
  llvm::StringRef getHashedPathSaved(llvm::ArrayRef<llvm::StringRef> path);
  std::vector<std::string> getPathParts(Node node);
  void maybeDumpTypes(Node node, TypeExpr *type);

  // Type variables don't have the same scoping rules as other variables.
  // When type variables in introduced, we're not sure what type we're
  // even declaring or what the scope is, and we don't want the declarees
  // to be in a more narrow scope. Type variables live only long enough to
  // declare the type(s).
  // TODO
  void pushTypeVariableScope() {}
  void popTypeVariableScope() {}

  // Environment for type variables
  Env env;
  Env typeEnv;

  // Set of types that have been declared as concrete, usually because they
  // are type variables for parameters of a function.
  ConcreteTypes concreteTypes;

  // Allocator for type expressions
  std::vector<std::unique_ptr<TypeExpr>> typeArena;

  // Paths to search for type variables
  llvm::SmallVector<llvm::StringRef> moduleSearchPath;

  // Current module path for code being inferred
  llvm::SmallVector<llvm::StringRef> currentModule;

  // It is useful to keep and dump certain nodes and types for debugging
  // and testing. Record them here to be dumped after inference is complete.
  llvm::SmallVector<std::string> nodesToDump;

  // Sidecar for caching inferred types
  llvm::DenseMap<ts::NodeID, TypeExpr *> nodeToType;

  // Sidecar for caching parameter descriptors when inferring function
  // argument types.
  llvm::DenseMap<ts::NodeID, ParameterDescriptor> parameterDescSidecar;

  // Record the record fields seen so far so we can infer the use of a field
  // on an uninferred variable can automatically promote to the last seen record
  // type with a matching field name.
  llvm::SmallVector<std::pair<llvm::StringRef, llvm::SmallVector<llvm::StringRef>>> seenRecordFields;

  // Root scope for the unifier, created when the unifier is constructed.
  // Sometimes we need to declare stdlib types before actually loading the
  // stdlib, in which case we don't have a compilation unit to create a scope.
  std::unique_ptr<detail::Scope> rootScope;

  // Whether we are loading the standard library. It's helpful to skip certain
  // debugging steps when loading the stdlib, as it becomes very noisy when
  // looking at inference of user code.
  bool isLoadingStdlib = false;

  // The maximum number of errors that can be reported.
  int maxErrors = 5;

  SmallVector<Diagnostic> diagnostics;
  nullptr_t error(std::string message, ts::Node node, const char* filename, unsigned long lineno);
  nullptr_t error(std::string message, const char* filename, unsigned long lineno);

  // RAII wrappers for pushing and popping search paths and environment scopes.
  friend struct detail::ModuleSearchPathScope;
  friend struct detail::ModuleScope;
  friend struct detail::Scope;
  friend struct detail::TypeVariableScope;
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
      : unifier(unifier), envScope(unifier->env), typeEnvScope(unifier->typeEnv),
        concreteTypes(unifier->concreteTypes) {}
  ~Scope() { unifier->concreteTypes = std::move(concreteTypes); }

private:
  Unifier *unifier;
  Unifier::EnvScope envScope;
  Unifier::EnvScope typeEnvScope;
  Unifier::ConcreteTypes concreteTypes;
};

struct TypeVariableScope {
  TypeVariableScope(Unifier &unifier) : unifier(unifier) {
    unifier.pushTypeVariableScope();
  }
  ~TypeVariableScope() { unifier.popTypeVariableScope(); }

private:
  Unifier &unifier;
};

} // namespace detail
} // namespace ocamlc2

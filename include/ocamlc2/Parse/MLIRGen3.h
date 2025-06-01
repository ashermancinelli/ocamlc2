#pragma once
#include "cpp-tree-sitter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include "ocamlc2/Dialect/OcamlTypeUtils.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <ocamlc2/Parse/ScopedHashTable.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include "ocamlc2/Parse/TSUnifier.h"

namespace ocamlc2 {

struct BuiltinBuilder {
  std::function<mlir::FailureOr<mlir::Value>(mlir::Location, mlir::ValueRange)> builder;
  mlir::FailureOr<mlir::Value> operator()(mlir::Location loc, mlir::ValueRange args) const {
    return builder(loc, args);
  }
};

struct MLIRGen3;
using VariableScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using ModuleSearchPathScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::ocaml::ModuleOp>;
using VariantDeclarations = std::vector<std::pair<std::string, std::optional<mlir::Type>>>;
using Callee = std::variant<mlir::func::FuncOp, mlir::Value, BuiltinBuilder>;

struct Scope;

struct MLIRGen3 {
  MLIRGen3(mlir::MLIRContext &context, Unifier &unifier, ts::Node root) 
    : context(context), builder(&context), unifier(unifier), root(root) {}
  mlir::FailureOr<mlir::OwningOpRef<mlir::ocaml::ModuleOp>> gen();
  mlir::FailureOr<mlir::Value> gen(const Node node);
  inline mlir::Location loc(const Node node) const {
    auto range = node.getPointRange();
    auto start = range.start, end = range.end;
    return mlir::FileLineColRange::get(
        mlir::StringAttr::get(&context,
                              unifier.sources.back().filepath.string()),
        start.row + 1, start.column + 1, end.row + 1, end.column + 1);
  }
  inline auto error(const mlir::Location loc) const {
    return mlir::emitError(loc) << "error: ";
  }
  inline auto error(const Node node) const {
    unifier.show(node.getCursor(), false);
    return error(loc(node));
  }
  template<typename... Args>
  inline auto nyi(const mlir::Location loc, Args&&... args) const {
    return error(loc) << "NYI: " << llvm::join(llvm::ArrayRef{args...}, ", ");
  }
  inline auto nyi(const Node node) const {
    unifier.show(node.getCursor(), false);
    return error(node) << "NYI: " << node.getType();
  }
  inline mlir::ocaml::ModuleOp getModule() const {
    return module.get();
  }
  inline Unifier &getUnifier() const {
    return unifier;
  }
private:
  mlir::FailureOr<mlir::Value> genLetBinding(const Node node);
  mlir::FailureOr<mlir::Value> genValueDefinition(const Node node);
  mlir::FailureOr<mlir::Value> genForExpression(const Node node);
  mlir::FailureOr<mlir::Value> genCompilationUnit(const Node node);
  mlir::FailureOr<mlir::Value> genNumber(const Node node);
  mlir::FailureOr<mlir::Value> genApplicationExpression(const Node node);
  mlir::FailureOr<mlir::Value> genCallOrCurry(mlir::Location loc, mlir::Value closure, llvm::ArrayRef<mlir::Value> args);
  mlir::FailureOr<mlir::Value> genApplication(mlir::Location, Callee callee, llvm::ArrayRef<mlir::Value> args);
  mlir::FailureOr<Callee> genBuiltinCallee(const Node node);
  mlir::FailureOr<Callee> genCallee(const Node node);
  mlir::FailureOr<mlir::Value> genInfixExpression(const Node node);
  mlir::FailureOr<mlir::Value> genLetExpression(const Node node);
  mlir::FailureOr<mlir::Value> genLetBindingValueDefinition(const Node patternNode, const Node bodyNode);
  mlir::FailureOr<mlir::Value> genMatchExpression(const Node node);
  mlir::FailureOr<mlir::Value> genConstructorPattern(const Node node);
  mlir::FailureOr<mlir::Value> genValuePattern(const Node node);
  mlir::FailureOr<mlir::Value> genSequenceExpression(const Node node);
  mlir::FailureOr<mlir::Value> genString(const Node node);
  mlir::FailureOr<mlir::Value> genIfExpression(const Node node);
  mlir::FailureOr<mlir::Value> genFunExpression(const Node node);
  mlir::FailureOr<mlir::Value> genListExpression(const Node node);
  mlir::FailureOr<mlir::Value> genConsExpression(const Node node);
  mlir::FailureOr<mlir::Value> genExternal(const Node node);
  mlir::FailureOr<mlir::Value> genModuleDefinition(const Node node);
  mlir::FailureOr<mlir::Value> genModuleStructure(const Node node);
  mlir::FailureOr<mlir::Value> genModuleBinding(const Node node);
  mlir::FailureOr<mlir::Value> genPrefixExpression(const Node node);
  mlir::FailureOr<mlir::func::FuncOp>
  genFunctionBody(llvm::StringRef name, mlir::FunctionType funType,
                  mlir::Location loc, llvm::ArrayRef<Node> parameters,
                  Node bodyNode);
  mlir::FailureOr<mlir::Value> genArrayGetExpression(const Node node);
  mlir::FailureOr<mlir::Value> genArrayExpression(const Node node);
  mlir::FailureOr<mlir::Value> genGlobalForFreeVariable(mlir::Value value,
                                                        llvm::StringRef name,
                                                        mlir::Location loc);

  mlir::FailureOr<mlir::Value> declareVariable(Node node, mlir::Value value,
                                               mlir::Location loc,
                                               VariableScope *scope = nullptr);
  mlir::FailureOr<mlir::Value> declareVariable(llvm::StringRef name,
                                               mlir::Value value,
                                               mlir::Location loc,
                                               VariableScope *scope = nullptr);
  llvm::SmallVector<mlir::ocaml::ModuleOp> getModuleSearchPath() const;
  mlir::FailureOr<mlir::Value> getVariable(llvm::StringRef name, mlir::Location loc);
  mlir::FailureOr<mlir::Value> getVariable(SmallVector<llvm::StringRef> path, mlir::Location loc);
  mlir::FailureOr<mlir::Operation*> getVariableInModule(mlir::ocaml::ModuleOp module, llvm::StringRef name, mlir::Location loc);
  mlir::FailureOr<mlir::Value> getVariable(const Node node);
  mlir::FailureOr<Callee> genConstructorPath(const Node node);

  mlir::FailureOr<mlir::Type> mlirType(ocamlc2::TypeExpr *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirType(ocamlc2::VariantOperator *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirType(const Node node);
  mlir::FailureOr<mlir::Type> mlirFunctionType(ocamlc2::TypeExpr *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirFunctionType(const Node node);
  mlir::FailureOr<mlir::Type> mlirVariantCtorType(ocamlc2::CtorOperator *ctor, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirTypeFromBasicTypeOperator(llvm::StringRef name);

  // If we find a variable at the top-level is shadowing others, we will only export the last use
  // of the identifier. When we encounter a reused identifier, we need to visit all the uses
  // of the identifier and mangle their names.
  mlir::LogicalResult shadowGlobalsIfNeeded(llvm::StringRef identifier);

  mlir::FailureOr<std::variant<mlir::ocaml::ProgramOp, mlir::func::FuncOp>>
  getCurrentFuncOrProgram(mlir::Operation *op=nullptr);
  bool shouldAddToModuleType(mlir::Operation *op);
  mlir::FailureOr<mlir::Value> findEnvForFunction(mlir::func::FuncOp funcOp);
  mlir::FailureOr<mlir::Value> findEnvForFunctionOrNullEnv(mlir::func::FuncOp funcOp);
  llvm::StringRef getText(const Node node);
  llvm::StringRef getTextFromValuePath(Node node);
  llvm::SmallVector<llvm::StringRef> getTextPathFromValuePath(Node node);
  llvm::StringRef getIdentifierTextFromPattern(const Node node);
  llvm::StringRef getTextStripQuotes(const Node node);
  inline auto *unifierType(const Node node) {
    return unifier.getInferredType(node);
  }

  void pushCaptureID(ts::NodeID id) {
    captureIDs.push_back(id);
  }
  void popCaptureID() {
    captureIDs.pop_back();
  }
  mlir::FailureOr<std::tuple<bool, mlir::func::FuncOp>>
  valueIsFreeInCurrentContext(mlir::Value value);

  llvm::SmallVector<mlir::ocaml::ModuleType> moduleTypeStack;
  mlir::ocaml::ModuleType getCurrentModuleType() const {
    return moduleTypeStack.back();
  }
  void pushModuleType(mlir::ocaml::ModuleType moduleType) {
    moduleTypeStack.push_back(moduleType);
  }
  void popModuleType() {
    moduleTypeStack.pop_back();
  }

  void pushModule(mlir::ocaml::ModuleOp module);
  mlir::ocaml::ModuleOp popModule();
  mlir::ocaml::ModuleOp getCurrentModule() const;
  mlir::ocaml::ModuleOp getRootModule() const;

  llvm::SmallVector<ts::NodeID> captureIDs;
  llvm::DenseMap<TypeExpr *, mlir::Type> typeExprToMlirType;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> variables;
  mlir::MLIRContext &context;
  mlir::ocaml::OcamlOpBuilder builder;
  Unifier &unifier;
  mlir::OwningOpRef<mlir::ocaml::ModuleOp> module;
  llvm::SmallVector<mlir::ocaml::ModuleOp> moduleStack;
  llvm::SmallVector<mlir::ocaml::ModuleOp> moduleSearchPath;
  ts::Node root;
  friend struct Scope;
};

} // namespace ocamlc2

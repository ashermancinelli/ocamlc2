#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include <ocamlc2/Parse/ScopedHashTable.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include "ocamlc2/Parse/TSUnifier.h"

namespace ocamlc2 {
struct MLIRGen3;
using VariableScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using VariantDeclarations = std::vector<std::pair<std::string, std::optional<mlir::Type>>>;

struct MLIRGen3 {
  MLIRGen3(mlir::MLIRContext &context, Unifier &unifier, ts::Node root) 
    : context(context), builder(&context), unifier(unifier), root(root) {}
  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen();
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
  inline auto nyi(const Node node) const {
    unifier.show(node.getCursor(), false);
    return error(node) << "NYI: " << node.getType();
  }
  inline mlir::ModuleOp getModule() const {
    return module.get();
  }
private:
  mlir::FailureOr<mlir::Value> genLetBinding(const Node node);
  mlir::FailureOr<mlir::Value> genValueDefinition(const Node node);
  mlir::FailureOr<mlir::Value> genForExpression(const Node node);
  mlir::FailureOr<mlir::Value> genCompilationUnit(const Node node);
  mlir::FailureOr<mlir::Value> genNumber(const Node node);
  mlir::FailureOr<mlir::Value> genApplicationExpression(const Node node);
  mlir::FailureOr<mlir::func::FuncOp> genCallee(const Node node);
  mlir::FailureOr<mlir::Value> genInfixExpression(const Node node);
  mlir::FailureOr<mlir::Value> genLetExpression(const Node node);
  mlir::FailureOr<mlir::Value> genLetBindingValueDefinition(const Node patternNode, const Node bodyNode);

  mlir::FailureOr<mlir::Value> declareVariable(Node node, mlir::Value value, mlir::Location loc);
  mlir::FailureOr<mlir::Value> declareVariable(llvm::StringRef name, mlir::Value value, mlir::Location loc);
  mlir::FailureOr<mlir::Value> getVariable(llvm::StringRef name, mlir::Location loc);
  mlir::FailureOr<mlir::Value> getVariable(const Node node);

  mlir::FailureOr<mlir::Type> mlirType(ocamlc2::TypeExpr *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirType(ocamlc2::VariantOperator *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirType(const Node node);
  mlir::FailureOr<mlir::Type> mlirFunctionType(ocamlc2::TypeExpr *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirFunctionType(const Node node);
  mlir::FailureOr<mlir::Type> mlirTypeFromBasicTypeOperator(llvm::StringRef name);
  llvm::StringRef getText(const Node node);
  inline auto *unifierType(const Node node) {
    return unifier.getInferredType(node);
  }
  llvm::DenseMap<TypeExpr *, mlir::Type> typeExprToMlirType;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> variables;
  mlir::MLIRContext &context;
  mlir::ocaml::OcamlOpBuilder builder;
  Unifier &unifier;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ts::Node root;
};
} // namespace ocamlc2

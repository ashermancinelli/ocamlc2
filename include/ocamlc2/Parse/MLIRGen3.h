#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include "ocamlc2/Parse/TSUnifier.h"

struct MLIRGen3;
using VariableScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using VariantDeclarations = std::vector<std::pair<std::string, std::optional<mlir::Type>>>;

struct MLIRGen3 {
  MLIRGen3(mlir::MLIRContext &context, ocamlc2::ts::Unifier &unifier, ts::Node root) 
    : context(context), builder(&context), unifier(unifier), root(root) {}
  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen();
  mlir::FailureOr<mlir::Value> gen(const ocamlc2::ts::Node node);
  inline mlir::Location loc(const ocamlc2::ts::Node node) const {
    auto range = node.getPointRange();
    auto start = range.start, end = range.end;
    return mlir::FileLineColRange::get(
        mlir::StringAttr::get(&context, unifier.filepath), start.row + 1,
        start.column + 1, end.row + 1, end.column + 1);
  }
  inline auto error(const mlir::Location loc) const {
    return mlir::emitError(loc) << "error: ";
  }
  inline auto error(const ocamlc2::ts::Node node) const {
    unifier.show(node.getCursor(), true);
    return error(loc(node));
  }
  inline auto nyi(const ocamlc2::ts::Node node) const {
    unifier.show(node.getCursor(), true);
    return error(node) << "not yet implemented: " << node.getType();
  }
  inline mlir::ModuleOp getModule() const {
    return module.get();
  }
private:
  mlir::FailureOr<mlir::Value> genLetBinding(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Value> genValueDefinition(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Value> genForExpression(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Value> genCompilationUnit(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Value> genNumber(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Value> genApplicationExpression(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::func::FuncOp> genCallee(const ocamlc2::ts::Node node);

  mlir::FailureOr<mlir::Value> declareVariable(ocamlc2::ts::Node node, mlir::Value value, mlir::Location loc);
  mlir::FailureOr<mlir::Value> declareVariable(llvm::StringRef name, mlir::Value value, mlir::Location loc);
  mlir::FailureOr<mlir::Value> getVariable(llvm::StringRef name, mlir::Location loc);

  mlir::FailureOr<mlir::Type> mlirType(const ocamlc2::TypeExpr *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirType(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Type> mlirFunctionType(const ocamlc2::TypeExpr *type, mlir::Location loc);
  mlir::FailureOr<mlir::Type> mlirFunctionType(const ocamlc2::ts::Node node);
  mlir::FailureOr<mlir::Type> mlirTypeFromBasicTypeOperator(llvm::StringRef name);
  llvm::StringRef getText(const ocamlc2::ts::Node node);
  inline auto *unifierType(const ocamlc2::ts::Node node) {
    return unifier.getInferredType(node);
  }
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> variables;
  mlir::MLIRContext &context;
  mlir::ocaml::OcamlOpBuilder builder;
  ocamlc2::ts::Unifier &unifier;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ts::Node root;
};

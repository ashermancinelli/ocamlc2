
#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ocamlc2/Dialect/OcamlOpBuilder.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/LogicalResult.h>

struct MLIRGen2;
struct TypeConstructor {
  using FunctionType = std::function<
    mlir::Type( /* Create a type */
      MLIRGen2 &, 
      llvm::ArrayRef<mlir::Type> /* With these type parameters */
    )
  >;
  FunctionType constructor;
};
using TypeConstructorScope = llvm::ScopedHashTableScope<llvm::StringRef, TypeConstructor>;
using VariableScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using VariantDeclarations = std::vector<std::pair<std::string, std::optional<mlir::Type>>>;

struct MLIRGen2 {
  MLIRGen2(mlir::MLIRContext &context, std::unique_ptr<ocamlc2::ASTNode> compilationUnit) 
    : context(context), builder(&context), compilationUnit(std::move(compilationUnit)) {}
  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen();
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ASTNode const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::NumberExprAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ConstructorPathAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::TypeConstructorPathAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ApplicationExprAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::CompilationUnitAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ExpressionItemAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ValueDefinitionAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::LetBindingAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ValuePathAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ForExpressionAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::LetExpressionAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::InfixExpressionAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::ParenthesizedExpressionAST const& node);

  mlir::FailureOr<mlir::Value> gen(ocamlc2::TypeDefinitionAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::TypeBindingAST const& node);
  mlir::FailureOr<std::optional<mlir::Type>> gen(ocamlc2::ConstructorDeclarationAST const& node);

  mlir::LogicalResult genVariantConstructors(mlir::ocaml::VariantType variantType, mlir::Location loc);
  mlir::FailureOr<VariantDeclarations> gen(ocamlc2::VariantDeclarationAST const& node);
  mlir::FailureOr<mlir::Value> gen(ocamlc2::MatchExpressionAST const& node);

  using MatchCases = std::vector<std::unique_ptr<ocamlc2::MatchCaseAST>>;
  mlir::FailureOr<mlir::Value> genMatchCases(
      MatchCases const& cases,
      mlir::Value scrutinee, mlir::Type resultType, mlir::Location location);
  mlir::FailureOr<mlir::Value> genMatchCase(
      MatchCases::const_iterator current, MatchCases::const_iterator end,
      mlir::Value scrutinee, mlir::Type resultType, mlir::Location location);

  mlir::FailureOr<mlir::Value> genPattern(ocamlc2::ASTNode const& node, mlir::Value scrutinee);
  mlir::FailureOr<mlir::Value> genPattern(ocamlc2::ConstructorPathAST const& node, mlir::Value scrutinee);
  mlir::FailureOr<mlir::Value> genPattern(ocamlc2::ValuePatternAST const& node, mlir::Value scrutinee);

  mlir::FailureOr<mlir::Value> declareVariable(llvm::StringRef name, mlir::Value value, mlir::Location loc);
  mlir::FailureOr<mlir::Value> getVariable(llvm::StringRef name, mlir::Location loc);

  mlir::LogicalResult declareTypeConstructor(llvm::StringRef name, TypeConstructor constructor, mlir::Location loc);
  mlir::FailureOr<TypeConstructor> getTypeConstructor(ocamlc2::ASTNode const& node);
  mlir::FailureOr<std::string> getApplicatorName(ocamlc2::ASTNode const& node);
  mlir::FailureOr<mlir::Value> genRuntime(llvm::StringRef name, ocamlc2::ApplicationExprAST const& node);

  // have to figure out the mlir type associated with the parameter name
  // if we are able to.
  mlir::FailureOr<std::pair<std::vector<std::string>, std::vector<mlir::Type>>>
  processParameters(
      std::vector<std::unique_ptr<ocamlc2::ASTNode>> const &parameters);

  void initializeTypeConstructors();
  inline mlir::Location loc(const ocamlc2::ASTNode *node) const {
    return node->getMLIRLocation(context);
  }
  inline mlir::ModuleOp getModule() const {
    return module.get();
  }
private:
  llvm::ScopedHashTable<llvm::StringRef, TypeConstructor> typeConstructors;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> variables;
  mlir::MLIRContext &context;
  mlir::ocaml::OcamlOpBuilder builder;
  std::unique_ptr<ocamlc2::ASTNode> compilationUnit;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

#pragma once
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Parse/TSAdaptor.h"
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "ocamlc2/Dialect/OcamlOpBuilder.h"

using Scope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using Node = std::pair<StringRef, TSNode>;
using Argument = std::pair<std::string, optional<std::string>>;
using NodeList = std::vector<Node>;
using NodeIter = NodeList::iterator;

struct MLIRGen;
struct TypeConstructor {
  std::function<mlir::Type(MLIRGen &)> constructor;
};
using TypeConstructorScope = llvm::ScopedHashTableScope<llvm::StringRef, TypeConstructor>;

struct MLIRGen {
  MLIRGen(mlir::MLIRContext &context);
  FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen(TSTreeAdaptor &&adaptor);
  void genCompilationUnit(TSNode node);
  FailureOr<mlir::Value> gen(NodeIter it);
  FailureOr<mlir::Value> gen(NodeList & nodes);
  FailureOr<mlir::Value> genLetBinding(TSNode node);
  FailureOr<mlir::Value> genAssign(StringRef lhs, mlir::Value rhs);
  LogicalResult declareValue(llvm::StringRef name, mlir::Value value);
  FailureOr<mlir::Value> genRuntimeCall(llvm::StringRef name, mlir::ValueRange args, mlir::Location loc, TSNode *node);
  FailureOr<mlir::func::FuncOp> lookupFunction(llvm::StringRef name);
  FailureOr<mlir::Value> lookupValuePath(TSNode *node);
  FailureOr<mlir::Value> lookupValue(llvm::StringRef name, std::optional<mlir::Location> maybeLoc=std::nullopt);
  FailureOr<std::string> valuePathToIdentifier(TSNode *node);
  FailureOr<std::vector<mlir::Type>> getPrintfTypeHints(mlir::ValueRange args, TSNode *stringContentNode);
  FailureOr<std::vector<Argument>> getFunctionArguments(NodeIter it);
  void insertBuiltinTypeConstructors();
  std::string getUniqueName(std::string_view prefix="");
  std::string mangleIdentifier(llvm::StringRef name);
  std::string sanitizeParsedString(TSNode *node);
  mlir::Location loc(TSNode node);
  inline mlir::ModuleOp getModule() const { return *module; }
  inline mlir::MLIRContext &getContext() const { return context; }
  inline mlir::ocaml::OcamlOpBuilder &getBuilder() { return builder; }
private:
  TSTreeAdaptor *adaptor;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  llvm::ScopedHashTable<llvm::StringRef, TypeConstructor> typeConstructors;
  mlir::MLIRContext &context;
  mlir::ocaml::OcamlOpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

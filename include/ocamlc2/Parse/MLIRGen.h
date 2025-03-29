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

using Scope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
using Node = std::pair<StringRef, TSNode>;
using NodeList = std::vector<Node>;
using NodeIter = NodeList::iterator;

struct RuntimeFunction {
  RuntimeFunction(
      llvm::StringRef name,
      std::function<void(mlir::OpBuilder &, mlir::ModuleOp)> genDeclareFunc,
      std::function<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                mlir::ValueRange)>
          genCallFunc)
      : name(name), genDeclareFunc(genDeclareFunc), genCallFunc(genCallFunc) {}
  llvm::StringRef name;
  std::function<void(mlir::OpBuilder &, mlir::ModuleOp)> genDeclareFunc;
  std::function<mlir::Value(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)> genCallFunc;
  void genDeclare(mlir::OpBuilder &builder, mlir::ModuleOp module) const {
    if (not declared) {
      genDeclareFunc(builder, module);
      declared = true;
    }
  }
  mlir::Value genCall(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) const {
    return genCallFunc(builder, loc, args);
  }
private:
  mutable bool declared = false;
};

class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder);
  FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> gen(TSTreeAdaptor &&adaptor);
  void genCompilationUnit(TSNode node);
  FailureOr<mlir::Value> gen(NodeIter it);
  FailureOr<mlir::Value> gen(NodeList & nodes);
  FailureOr<mlir::Value> genLetBinding(TSNode node);
  FailureOr<mlir::Value> genAssign(StringRef lhs, mlir::Value rhs);
  LogicalResult declareFunction(llvm::StringRef name, mlir::FunctionType type);
  LogicalResult declareValue(llvm::StringRef name, mlir::Value value);
  FailureOr<mlir::Value> genRuntimeCall(llvm::StringRef name, mlir::ValueRange args, mlir::Location loc);
  FailureOr<mlir::FunctionType> lookupFunction(llvm::StringRef name);
  FailureOr<mlir::Value> lookupValuePath(TSNode *node);
  FailureOr<mlir::Value> lookupValue(llvm::StringRef name, std::optional<mlir::Location> maybeLoc=std::nullopt);
  FailureOr<std::string> valuePathToIdentifier(TSNode *node);
  std::string getUniqueName(std::string_view prefix="");
  std::string mangleIdentifier(llvm::StringRef name);
  std::string sanitizeParsedString(TSNode *node);
  mlir::Location loc(TSNode node);
  inline mlir::MLIRContext &getContext() const { return context; }
private:
  TSTreeAdaptor *adaptor;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  llvm::ScopedHashTable<llvm::StringRef, mlir::FunctionType> functionTable;
  mlir::MLIRContext &context;
  mlir::OpBuilder &builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

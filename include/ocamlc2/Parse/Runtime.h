#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <llvm/ADT/ArrayRef.h>
class MLIRGen;
struct TSNode;

struct RuntimeFunction {
  llvm::StringRef name;
  std::function<mlir::Value(MLIRGen *gen, TSNode *node, mlir::Location loc, mlir::ValueRange args)> call;
  static llvm::ArrayRef<RuntimeFunction> getRuntimeFunctions();
};


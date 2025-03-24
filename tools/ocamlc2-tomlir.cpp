#include <iostream>
#include <cstdint>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "ocamlc2/Parse/TSAdaptor.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Parse/MLIRGen.h"
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <ocaml-file>" << std::endl;
    return 1;
  }

  fs::path filepath = argv[1];
  assert(fs::exists(filepath) && "File does not exist");
  std::string source = must(slurpFile(filepath));
  TSTreeAdaptor tree(filepath.string(), source);

  // Create and configure an MLIRContext
  mlir::MLIRContext context;
  
  // Register the dialects we need
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // Create the IR builder
  mlir::OpBuilder builder(&context);
  llvm::outs() << "OCaml source: " << source << "\n";

  MLIRGen gen(context, builder);
  auto maybeModule = gen.gen(std::move(tree));
  if (failed(maybeModule)) {
    llvm::errs() << "Failed to generate MLIR\n";
    return 1;
  }
  auto &module = *maybeModule;

  // Create the top-level module operation
  // mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  
  // Set the insertion point to the module body
  
  // Create a function signature: () -> i32
  mlir::Type returnType = builder.getI32Type();
  mlir::FunctionType funcType = builder.getFunctionType({}, returnType);
  
  // Create the function
  mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);
  
  // Create the entry block
  mlir::Block &entryBlock = funcOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);
  
  // Create constant 0
  mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), 0, 32);
  
  // Return the zero value
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), zero);
  
  // Create a pass manager
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  
  // Apply the passes
  if (mlir::failed(pm.run(module.get()))) {
    llvm::errs() << "Failed to apply passes\n";
    return 1;
  }
  
  llvm::outs() << *module << "\n";
  
  return 0;
}

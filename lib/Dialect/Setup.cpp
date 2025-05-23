#include "ocamlc2/Dialect/OcamlDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ocamlc2/Dialect/OcamlPasses.h"
#include <mlir/Transforms/Passes.h>

namespace mlir::ocaml {
  void setupRegistry(mlir::DialectRegistry &registry) {
    mlir::func::registerAllExtensions(registry);
    mlir::LLVM::registerInlinerInterface(registry);

    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::ocaml::OcamlDialect>();
  }
  void setupContext(mlir::MLIRContext &context) {
    context.printOpOnDiagnostic(true);
    context.getOrLoadDialect<mlir::ocaml::OcamlDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
  }
  void setupDefaultPipeline(mlir::PassManager &pm) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::ocaml::createTypeInference());
    pm.addPass(mlir::ocaml::createLowerOCamlRuntime());
    pm.addPass(mlir::ocaml::createBufferizeBoxes());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
  void setupCodegenPipeline(mlir::PassManager &pm) {
    pm.addPass(mlir::ocaml::createConvertOCamlToLLVM());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
  }
}

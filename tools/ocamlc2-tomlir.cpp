#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"

#include "ocamlc2/Support/Logging.h"

#include "ocamlc2/Parse/TSAdaptor.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Parse/MLIRGen.h"
#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Dialect/OcamlPasses.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/CL.h"

#include <filesystem>
#include <iostream>
#include <cstdint>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <unistd.h>

#define DEBUG_TYPE "driver"
#include "ocamlc2/Support/Debug.h.inc"

namespace fs = std::filesystem;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ocaml file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ocamlc2 Ocaml compiler\n");

  if (ocamlc2::CL::RunGDB) {
    DBGS("Running under gdb\n");
    std::vector<char*> newArgs;
    const char *debugger = "lldb";
    newArgs.push_back(const_cast<char*>(debugger));
    newArgs.push_back(const_cast<char*>("--"));
    newArgs.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-gdb" or arg == "--gdb") {
        continue;
      }
      newArgs.push_back(argv[i]);
    }
    newArgs.push_back(nullptr);
    execvp(debugger, newArgs.data());
    std::cerr << "Failed to execute debugger: " << strerror(errno) << std::endl;
    return 1;
  }

  fs::path filepath = inputFilename.getValue();
  assert(fs::exists(filepath) && "File does not exist");
  std::string source = must(ocamlc2::slurpFile(filepath));
  TSTreeAdaptor tree(filepath.string(), source);
  DBGS("OCaml parsed:\n" << tree << "\n");

  // Create and configure an MLIRContext
  mlir::MLIRContext context;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceManagerHandler(sourceMgr, &context);

  mlir::ocaml::registerPasses();
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::ocaml::OcamlDialect>();
  mlir::ocaml::setupRegistry(registry);
  mlir::ocaml::setupContext(context);
  context.appendDialectRegistry(registry);
  context.printOpOnDiagnostic(true);
  context.getOrLoadDialect<mlir::ocaml::OcamlDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();

  // Create the IR builder
  DBGS("OCaml source:\n" << source << "\n");

  MLIRGen gen(context);
  auto maybeModule = gen.gen(std::move(tree));
  if (failed(maybeModule)) {
    DBGS("Failed to generate MLIR\n");
    return 1;
  }
  auto &module = *maybeModule;
  if (mlir::failed(module->verify())) {
    llvm::errs() << "Failed to verify MLIR\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  if (mlir::failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Failed to apply pass manager options\n";
    return 1;
  }

  // Optimization
  {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::ocaml::createLowerOCamlRuntime());
    pm.addPass(mlir::ocaml::createBufferizeBoxes());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  // Code generation
  {
    pm.addPass(mlir::ocaml::createConvertOCamlToLLVM());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
  }

  if (mlir::failed(pm.run(module.get()))) {
    llvm::errs() << "Failed to run pass manager\n";
    return 1;
  }

  llvm::outs() << *module << "\n";
}

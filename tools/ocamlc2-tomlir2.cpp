#include "ocamlc2/Dialect/OcamlPasses.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/MLIRGen2.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/CL.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>
#include <filesystem>
#include <llvm/Support/SourceMgr.h>
#include <memory>
#include <llvm/Support/CommandLine.h>
#include <mlir/IR/AsmState.h>
#define DEBUG_TYPE "ocamlc2-tomlir2"
#include "ocamlc2/Support/Debug.h.inc"

namespace fs = std::filesystem;

using namespace llvm;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ocaml file>"),
                                          cl::init("-"),
                                          cl::Required,
                                          cl::value_desc("filename"));

int main(int argc, char* argv[]) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "ocamlc2-tomlir2");
  TRACE();
  maybeReplaceWithGDB(argc, argv);
  fs::path filepath = inputFilename.getValue();
  std::string source = must(slurpFile(filepath));
  auto ast = ocamlc2::parse(source);
  DBGS("AST:\n" << *ast << "\n");

  mlir::MLIRContext context;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceManagerHandler(sourceMgr, &context);

  mlir::ocaml::registerPasses();
  mlir::DialectRegistry registry;
  mlir::ocaml::setupRegistry(registry);
  context.appendDialectRegistry(registry);
  mlir::ocaml::setupContext(context);

  MLIRGen2 gen(context, std::move(ast));
  auto module = gen.gen();

  if (mlir::failed(module)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    return 1;
  }

  DBGS("Module:\n" << module->get() << "\n");

  mlir::PassManager pm(&context);
  mlir::ocaml::setupDefaultPipeline(pm);
  mlir::ocaml::setupCodegenPipeline(pm);
  if (mlir::failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Failed to apply pass manager options\n";
    return 1;
  }

  if (mlir::failed(pm.run(module->get()))) {
    DBGS("Failed to run pass manager\n");
    return 1;
  }

  DBGS("Module:\n");
  llvm::outs() << module->get() << "\n";
  return 0;
}

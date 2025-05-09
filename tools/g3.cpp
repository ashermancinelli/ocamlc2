#include "ocamlc2/Dialect/OcamlDialect.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Parse/MLIRGen3.h"
#include "ocamlc2/Dialect/OcamlPasses.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/CL.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>
#include <filesystem>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <memory>
#include <llvm/Support/CommandLine.h>
#include <mlir/IR/AsmState.h>
#include <tree_sitter/tree-sitter-ocaml.h>
#include <cpp-tree-sitter.h>
#include <tree_sitter/api.h>
#define DEBUG_TYPE "g3"
#include "ocamlc2/Support/Debug.h.inc"

namespace fs = std::filesystem;

using namespace llvm;
using namespace ocamlc2;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ocaml file>"),
                                          cl::init("-"),
                                          cl::Required,
                                          cl::value_desc("filename"));

static cl::opt<bool> dumpIR("dump-camlir", cl::desc("Dump OCaml IR"), cl::init(false));

int main(int argc, char* argv[]) {
  fs::path exe = llvm::sys::fs::getMainExecutable(argv[0], nullptr);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "g3");
  TRACE();
  CL::maybeReplaceWithGDB(argc, argv);
  fs::path filepath = inputFilename.getValue();
  std::string source = must(ocamlc2::slurpFile(filepath));
  DBGS("Source:\n" << source << "\n");
  
  ocamlc2::Unifier unifier;
  if (failed(unifier.loadStdlibInterfaces(exe))) {
    llvm::errs() << "Failed to load stdlib interfaces\n";
    return 1;
  }
  if (failed(unifier.loadSourceFile(filepath))) {
    llvm::errs() << "Failed to load source file\n";
    return 1;
  }
  auto root = unifier.sources.back().tree.getRootNode();
  DBG(
    llvm::errs() << "AST:\n";
    unifier.show(root.getCursor(), true);
  )
  auto *te = unifier.infer(root.getCursor());
  DBGS("Inferred type: " << *te << '\n');

  mlir::MLIRContext context;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceManagerHandler(sourceMgr, &context);

  mlir::ocaml::registerPasses();
  mlir::DialectRegistry registry;
  mlir::ocaml::setupRegistry(registry);
  context.appendDialectRegistry(registry);
  mlir::ocaml::setupContext(context);

  MLIRGen3 gen(context, unifier, root);
  auto module = gen.gen();

  if (mlir::failed(module)) {
    DBGS("Failed to generate MLIR for compilation unit\n");
    return 1;
  }
  auto moduleOp = module->get();

  if (mlir::failed(moduleOp.verify())) {
    DBGS("Failed to verify module\n");
    moduleOp.print(llvm::errs());
    return 1;
  }
  DBG(llvm::dbgs() << "Module:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() << "\n";);

  if (dumpIR) {
    llvm::outs() << moduleOp << "\n";
  }

#if 0
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
#endif
  return 0;
}

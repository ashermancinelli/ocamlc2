#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/CL.h"
#include <iostream>
#include <filesystem>
#include <memory>
#include <llvm/Support/CommandLine.h>
#define DEBUG_TYPE "ocamlc2-parse2"
#include "ocamlc2/Support/Debug.h.inc"

namespace fs = std::filesystem;

using namespace llvm;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ocaml file>"),
                                          cl::init("-"),
                                          cl::Required,
                                          cl::value_desc("filename"));

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "ocamlc2-parse2");
  TRACE();
  fs::path filepath = inputFilename.getValue();
  std::string source = must(slurpFile(filepath));
  DBGS("Source:\n" << source << "\n");
  auto ast = ocamlc2::parse(source, filepath.string());
  DBGS("AST:\n" << *ast << "\n");
  
  // Perform type inference
  llvm::outs() << "Performing type inference...\n";
  auto typeScheme = ocamlc2::inferProgramType(ast.get());
  
  // Print the inferred type
  llvm::outs() << "Inferred type: ";
  ocamlc2::dumpTypeScheme(typeScheme);
  
  return 0;
}

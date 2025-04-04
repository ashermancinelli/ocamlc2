#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/MLIRGen2.h"
#include "ocamlc2/Support/Utils.h"
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
  llvm::outs() << "Debug: " << Debug << "\n";
  TRACE();
  fs::path filepath = inputFilename.getValue();
  std::string source = must(slurpFile(filepath));
  auto ast = ocamlc2::parse(source);
  DBGS("AST:\n" << *ast << "\n");
  return 0;
}

#include "ocamlc2/CamlParse/AST.h"
#include "ocamlc2/CamlParse/Parse.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/CL.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <llvm/Support/CommandLine.h>
#define DEBUG_TYPE "ocamlc2-camlparse"
#include "ocamlc2/Support/Debug.h.inc"

namespace fs = std::filesystem;
using namespace ocamlc2;

using namespace llvm;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ocaml file>"),
                                          cl::init("-"),
                                          cl::Required,
                                          cl::value_desc("filename"));

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "ocamlc2-camlparse");
  TRACE();
  fs::path filepath = inputFilename.getValue();
  std::ifstream file(filepath);
  auto ast = ocamlc2::parse(file, filepath.string());
  assert(ast && "Failed to parse file");
  llvm::outs() << *ast << "\n";
  
  return 0;
}

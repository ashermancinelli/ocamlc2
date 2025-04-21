#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/CL.h"
#include <iostream>
#include <filesystem>
#include <memory>
#include <llvm/Support/CommandLine.h>
#include <cpp-tree-sitter.h>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-ocaml.h>

#define DEBUG_TYPE "ocamlc2-p3"
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
  llvm::cl::ParseCommandLineOptions(argc, argv, "ocamlc2-p3");
  TRACE();
  fs::path filepath = inputFilename.getValue();
  std::string source = must(slurpFile(filepath));
  DBGS("Source:\n" << source << "\n");

  ::ts::Language language = tree_sitter_ocaml();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  auto root = tree.getRootNode();
  
  ocamlc2::ts::Unifier unifier{source};
  llvm::errs() << "AST:\n";
  unifier.show(root.getCursor(), true);
  auto *te = unifier.infer(root.getCursor());
  llvm::errs() << "Inferred type: " << *te << '\n';

  return 0;
}

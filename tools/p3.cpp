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
static cl::list<std::string> inputFilenames(cl::Positional,
                                          cl::desc("<input ocaml file name>"),
                                          cl::Required,
                                          cl::OneOrMore,
                                          cl::value_desc("filename"));

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "p3");
  TRACE();
  ocamlc2::ts::Unifier unifier;
  for (auto &filepath : inputFilenames) {
    unifier.loadSourceFile(filepath);
    DBG(llvm::errs() << "AST:\n"; unifier.show(
        unifier.sources.back().tree.getRootNode().getCursor(), true);)
    auto rootNode = unifier.sources.back().tree.getRootNode();
    auto *te = unifier.getType(rootNode.getID());
    DBGS("Inferred type: " << *te << '\n');
  }

  return 0;
}

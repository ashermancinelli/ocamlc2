#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Support/Repl.h"
#include <iostream>
#include <filesystem>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/FileSystem.h>
#include <memory>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
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
                                            cl::ZeroOrMore,
                                            cl::value_desc("filename"),
                                            cl::cat(ocamlc2::CL::OcamlOptions));

int main(int argc, char* argv[]) {
  fs::path exe = llvm::sys::fs::getMainExecutable(argv[0], nullptr);
  llvm::cl::ParseCommandLineOptions(argc, argv, "p3");
  llvm::EnablePrettyStackTrace();
  TRACE();
  CL::maybeReplaceWithGDB(argc, argv);
  TRACE();
  ocamlc2::Unifier unifier;
  TRACE();
  if (failed(unifier.loadStdlibInterfaces(exe))) {
    llvm::errs() << "Failed to load stdlib interfaces\n";
    return 1;
  }
  TRACE();
  auto set = std::set<std::string>(inputFilenames.begin(), inputFilenames.end());
  if (set.size() != inputFilenames.size()) {
    llvm::errs() << "Duplicate input file(s)\n";
    return 1;
  }
  for (auto &filepath : inputFilenames) {
    DBGS("Loading file: " << filepath << '\n');
    if (failed(unifier.loadSourceFile(filepath))) {
      unifier.showErrors();
      return 1;
    }
    if (CL::DParseTree) {
      unifier.showParseTree();
    }
    if (CL::DTypedTree) {
      unifier.showTypedTree();
    }
    DBG(llvm::errs() << "AST:\n"; unifier.show(true);)
    auto rootNode = unifier.sources.back().tree.getRootNode();
    auto *te = unifier.getType(rootNode.getID());
    DBGS("Inferred type: " << *te << '\n');
  }
  if (CL::Repl or CL::InRLWrap or inputFilenames.empty()) {
    runRepl(argc, argv, exe, unifier);
  }
  if (CL::DumpTypes) {
    unifier.dumpTypes(llvm::outs());
  }

  return 0;
}

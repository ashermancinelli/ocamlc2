#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cassert>
#include <filesystem>
#include "ocamlc2/Parse/TSAdaptor.h"
#include <tree_sitter/tree-sitter-ocaml.h>
#include <tree_sitter/api.h>
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Support/Utils.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <ocaml-file>" << std::endl;
    return 1;
  }
  
  fs::path filepath = argv[1];
  assert(fs::exists(filepath) && "File does not exist");
  std::string source = must(slurpFile(filepath));
  TSTreeAdaptor tree(filepath.string(), source);

  llvm::outs() << "Syntax tree for " << filepath << ":" << "\n" << tree << "\n";

  return 0;
}

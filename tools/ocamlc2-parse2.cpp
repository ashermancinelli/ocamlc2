#include "ocamlc2/Parse/MLIRGen.h"
#include "ocamlc2/Parse/AST.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <ocaml-file>" << std::endl;
    return 1;
  }
  fs::path filepath = argv[1];
  std::string source = must(slurpFile(filepath));
  auto ast = ocamlc2::parse(source);
  llvm::outs() << "AST: " << *ast << "\n";
  return 0;
}

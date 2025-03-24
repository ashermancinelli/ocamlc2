#include "ocamlc2/Parse/TSAdaptor.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-ocaml.h>

FailureOr<TSTree *> parseOCaml(const std::string &source) {
    return failure();
}

FailureOr<std::string> slurpFile(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    exit(1);
  }
  
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  return contents;
}

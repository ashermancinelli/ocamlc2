#include "ocamlc2/Parse/TSAdaptor.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-ocaml.h>

extern "C" const TSLanguage* tree_sitter_ocaml();

FailureOr<TSTree *> parseOCaml(const std::string &source) {
  TSParser *parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_ocaml());
  TSTree *tree = ts_parser_parse_string(parser, nullptr, source.c_str(), source.length());
  ts_parser_delete(parser);
  return tree;
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

void print_node(llvm::raw_ostream &os, TSNode node, std::string source, int indent = 0) {
  std::string indentation = "";
  for (int i = 0; i < indent * 2; ++i) {
    indentation += " ";
  }
  
  const char* node_type = ts_node_type(node);
  uint32_t start_byte = ts_node_start_byte(node);
  uint32_t end_byte = ts_node_end_byte(node);
  std::string text = source.substr(start_byte, end_byte - start_byte);
  
  os << indentation << node_type << ": \"" << text << "\"" << "\n";
  
  uint32_t child_count = ts_node_child_count(node);
  for (uint32_t i = 0; i < child_count; ++i) {
    TSNode child = ts_node_child(node, i);
    print_node(os, child, source, indent + 1);
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TSTreeAdaptor &adaptor) {
  auto walker = [&](TSNode node) {
    const char *node_type = ts_node_type(node);
    uint32_t start_byte = ts_node_start_byte(node);
    uint32_t end_byte = ts_node_end_byte(node);
    std::string text = adaptor.source.substr(start_byte, end_byte - start_byte);
    os << node_type << ": \"" << text << "\"" << "\n";
    return true;
  };
  (void)walker;
  print_node(os, ts_tree_root_node(adaptor.tree), adaptor.source);
  return os;
}

void TSTreeAdaptor::walk(Walker callback) const {
  walk("", callback);
}

void TSTreeAdaptor::walk(StringRef node_type, Walker callback) const {
  TSNode node = ts_tree_root_node(tree);
  unsigned childcount = ts_node_child_count(node);
  for (unsigned i = 0; i < childcount; ++i) {
    TSNode child = ts_node_child(node, i);
    if (node_type == "" or ts_node_type(child) == node_type) {
      if (not callback(child)) {
        return;
      }
    }
    walkRecurse(child, node_type, callback);
  }
}

bool TSTreeAdaptor::walkRecurse(TSNode node, StringRef node_type, Walker callback) const {
  if (node_type == "" or ts_node_type(node) == node_type) {
    if (not callback(node)) {
      return false;
    }
  }
  unsigned childcount = ts_node_child_count(node);
  for (unsigned i = 0; i < childcount; ++i) {
    TSNode child = ts_node_child(node, i);
    if (not walkRecurse(child, node_type, callback)) {
      return false;
    }
  }
  return true;
}

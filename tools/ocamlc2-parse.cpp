#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <tree_sitter/tree-sitter-ocaml.h>
#include <tree_sitter/api.h>

// Declare the tree-sitter parser for OCaml
extern "C" {
  const TSLanguage* tree_sitter_ocaml();
}

// Read a file into a string
std::string read_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
    exit(1);
  }
  
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  return contents;
}

// Print the syntax tree recursively
void print_node(TSNode node, std::string source, int indent = 0) {
  std::string indentation(indent * 2, ' ');
  
  // Get node type
  const char* node_type = ts_node_type(node);
  
  // Get node text
  uint32_t start_byte = ts_node_start_byte(node);
  uint32_t end_byte = ts_node_end_byte(node);
  std::string text = source.substr(start_byte, end_byte - start_byte);
  
  // Print node info
  std::cout << indentation << node_type << ": \"" << text << "\"" << std::endl;
  
  // Print children
  uint32_t child_count = ts_node_child_count(node);
  for (uint32_t i = 0; i < child_count; ++i) {
    TSNode child = ts_node_child(node, i);
    print_node(child, source, indent + 1);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <ocaml-file>" << std::endl;
    return 1;
  }
  
  // Read the input file
  std::string filepath = argv[1];
  std::string source = read_file(filepath);
  
  // Initialize tree-sitter parser
  TSParser* parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_ocaml());
  
  // Parse the source code
  TSTree* tree = ts_parser_parse_string(parser, nullptr, source.c_str(), source.length());
  
  // Get the root node
  TSNode root_node = ts_tree_root_node(tree);
  
  // Dump the syntax tree
  std::cout << "Syntax tree for " << filepath << ":" << std::endl;
  print_node(root_node, source);
  
  // Clean up
  ts_tree_delete(tree);
  ts_parser_delete(parser);
  
  return 0;
}

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/TypeSystem.h"
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TSPoint &point) {
  return os << point.row << ":" << point.column;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ts::Extent<ts::Point> &extent) {
  return os << extent.start << "-" << extent.end;
}

static void dump(ts::Cursor cursor, std::string_view source, unsigned indent = 0) {
  auto range = cursor.getCurrentNode().getPointRange();
  auto byteRange = cursor.getCurrentNode().getByteRange();
  std::string indentStr;
  for (unsigned i = 0; i < indent; ++i) {
    indentStr += std::string(ANSIColors::faint()) + "| " + ANSIColors::reset();
  }
  llvm::errs() << indentStr << ANSIColors::cyan() << cursor.getCurrentNode().getType() << ANSIColors::reset() << ": ";
  auto text = source.substr(byteRange.start, byteRange.end - byteRange.start);
  llvm::errs() << ANSIColors::italic() << ANSIColors::faint();
  if (text.contains('\n')) {
    llvm::errs() << "...";
  } else {
    llvm::errs() << text;
  }
  llvm::errs() << " " << range << "\n";
  llvm::errs() << ANSIColors::reset();
  auto node = cursor.getCurrentNode();
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    dump(child.getCursor(), source, indent + 1);
  }
  if (cursor.gotoNextSibling()) {
    dump(cursor.copy(), source, indent);
  }
}

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "ocamlc2-p3");
  TRACE();
  fs::path filepath = inputFilename.getValue();
  std::string source = must(slurpFile(filepath));
  DBGS("Source:\n" << source << "\n");

  ts::Language language = tree_sitter_ocaml();
  ts::Parser parser{language};
  auto tree = parser.parseString(source);
  auto root = tree.getRootNode();
  
  // Use the new pretty printer
  llvm::errs() << "AST:\n";
  dump(root.getCursor(), source);
  
  return 0;
}

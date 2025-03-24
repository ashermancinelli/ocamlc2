#pragma once
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "ocamlc2/Support/LLVMCommon.h"

struct TSTree;
struct TSNode;
extern "C" void ts_tree_delete(TSTree *tree);
FailureOr<TSTree *> parseOCaml(const std::string &source);
FailureOr<std::string> slurpFile(const std::string &path);
using Walker = std::function<bool(TSNode)>;

struct TSTreeAdaptor {
  TSTreeAdaptor(std::string filename, const std::string &source);
  StringRef getFilename() const { return filename; }
  StringRef getSource() const { return source; }
  void walk(Walker callback) const;
  void walk(StringRef node_type, Walker callback) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TSTreeAdaptor &adaptor);
  operator TSTree*() const { return tree.get(); }
  TSTreeAdaptor(TSTreeAdaptor &&other) noexcept;
private:
  bool walkRecurse(TSNode node, StringRef node_type, Walker callback) const;
  const std::string filename;
  const std::string &source;
  std::unique_ptr<TSTree, void(*)(TSTree*)> tree;
};

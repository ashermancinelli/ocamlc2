#pragma once

#include <llvm/Support/raw_ostream.h>
#include <cpp-tree-sitter.h>

namespace ocamlc2 {
inline namespace ts {
using namespace ::ts;
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TSPoint &point);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Extent<Point> &extent);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Cursor &cursor);
llvm::raw_ostream &
dump(llvm::raw_ostream &os, ts::Cursor cursor, std::string_view source,
     unsigned indent = 0, bool showUnnamed = false,
     std::optional<std::function<void(llvm::raw_ostream &, ts::Node)>> dumpNode=std::nullopt);
std::string_view getText(const ts::Node &node, std::string_view source);
llvm::SmallVector<ts::Node> getChildren(const ts::Node &node);
llvm::SmallVector<ts::Node> getNamedChildren(const ts::Node &node);
bool isLetBindingRecursive(Cursor ast);

} // namespace ts
} // namespace ocamlc2

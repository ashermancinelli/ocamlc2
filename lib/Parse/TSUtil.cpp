#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/Colors.h"
#include <llvm/Support/raw_ostream.h>
#include <cpp-tree-sitter.h>

namespace ocamlc2 {
inline namespace ts {
using namespace ::ts;
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TSPoint &point) {
  return os << point.row << ":" << point.column;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Extent<Point> &extent) {
  return os << extent.start << "-" << extent.end;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Cursor &cursor) {
  return os << cursor.getCurrentNode().getType();
}

std::string_view getText(const ts::Node &node, std::string_view source) {
  auto byteRange = node.getByteRange();
  return source.substr(byteRange.start, byteRange.end - byteRange.start);
}

llvm::raw_ostream &dump(llvm::raw_ostream &os, ts::Cursor cursor, std::string_view source, unsigned indent, bool showUnnamed) {
  auto range = cursor.getCurrentNode().getPointRange();
  auto byteRange = cursor.getCurrentNode().getByteRange();
  std::string indentStr;
  for (unsigned i = 0; i < indent; ++i) {
    indentStr += std::string(ANSIColors::faint()) + "| " + ANSIColors::reset();
  }
  os << indentStr << ANSIColors::cyan() << cursor.getCurrentNode().getType() << ANSIColors::reset() << ": ";
  auto text = source.substr(byteRange.start, byteRange.end - byteRange.start);
  os << ANSIColors::italic() << ANSIColors::faint();
  if (text.contains('\n')) {
    os << "...";
  } else {
    os << text;
  }
  os << " " << range << "\n";
  os << ANSIColors::reset();
  auto node = cursor.getCurrentNode();
  if (showUnnamed) {
    for (unsigned i = 0; i < node.getNumChildren(); ++i) {
      auto child = node.getChild(i);
      dump(os, child.getCursor(), source, indent + 1);
    }
  } else {
    for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
      auto child = node.getNamedChild(i);
      dump(os, child.getCursor(), source, indent + 1);
    }
  }
  if (cursor.gotoNextSibling()) {
    dump(os, cursor.copy(), source, indent);
  }
  return os;
}

} // namespace ts
} // namespace ocamlc2

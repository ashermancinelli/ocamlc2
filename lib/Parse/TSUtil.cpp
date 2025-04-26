#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/Colors.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <cpp-tree-sitter.h>

#define DEBUG_TYPE "tsutil"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ts;

namespace ocamlc2 {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TSPoint point) {
  return os << point.row << ":" << point.column;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Extent<Point> extent) {
  return os << extent.start << "-" << extent.end;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Cursor cursor) {
  return os << cursor.getCurrentNode().getType();
}

std::string_view getText(const ts::Node &node, std::string_view source) {
  auto byteRange = node.getByteRange();
  return source.substr(byteRange.start, byteRange.end - byteRange.start);
}

static std::string getIndent(unsigned indent) {
  std::string indentStr;
  for (unsigned i = 0; i < indent; ++i) {
    indentStr += std::string(ANSIColors::faint()) + "| " + ANSIColors::reset();
  }
  return indentStr;
}

llvm::raw_ostream &
dump(llvm::raw_ostream &os, ts::Cursor cursor, std::string_view source,
     unsigned indent, bool showUnnamed,
     std::optional<std::function<void(llvm::raw_ostream &, ts::Node)>> dumpNode) {
  auto range = cursor.getCurrentNode().getPointRange();
  auto byteRange = cursor.getCurrentNode().getByteRange();
  os << getIndent(indent) << ANSIColors::cyan() << cursor.getCurrentNode().getType() << ANSIColors::reset() << ": ";
  auto text = source.substr(byteRange.start, byteRange.end - byteRange.start);
  os << ANSIColors::italic() << ANSIColors::faint();
  if (text.contains('\n')) {
    os << "...";
  } else {
    os << text;
  }
  os << " " << range;
  if (dumpNode) {
    auto callback = *dumpNode;
    callback(os, cursor.getCurrentNode());
  }
  os << "\n";
  os << ANSIColors::reset();
  auto node = cursor.getCurrentNode();
  auto showLabel = [&] (auto node, unsigned index) {
    auto name = showUnnamed ? node.getFieldNameForChild(index) : node.getFieldNameForNamedChild(index);
    if (!name.empty()) {
      os << getIndent(indent + 1) << ANSIColors::faint() << ANSIColors::italic() << name
         << ": " << ANSIColors::reset() << '\n';
    }
  };
  auto children = showUnnamed ? getChildren(node) : getNamedChildren(node);
  for (auto [i, child] : llvm::enumerate(children)) {
    showLabel(node, i);
    dump(os, child.getCursor(), source, indent + 1, showUnnamed, dumpNode);
  }
  if (cursor.gotoNextSibling()) {
    dump(os, cursor.copy(), source, indent, showUnnamed, dumpNode);
  }
  return os;
}

llvm::SmallVector<ts::Node> getChildren(const ts::Node node) {
  llvm::SmallVector<ts::Node> children;
  for (unsigned i = 0; i < node.getNumChildren(); ++i) {
    children.push_back(node.getChild(i));
  }
  return children;
}

llvm::SmallVector<ts::Node> getNamedChildren(const ts::Node node) {
  llvm::SmallVector<ts::Node> children;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    children.push_back(node.getNamedChild(i));
  }
  return children;
}

llvm::SmallVector<ts::Node> getArguments(const ts::Node node) {
  TRACE();
  assert(node.getType() == "application_expression" && "expecting application_expression");
  llvm::SmallVector<ts::Node> children;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    if (node.getFieldNameForNamedChild(i) == "argument") {
      DBGS("child: " << child.getSExpr().get() << '\n');
      children.push_back(child);
    }
  }
  return children;
}

bool isLetBindingRecursive(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "let_binding" && "expecting let_binding");
  auto maybeRecKeyword = node.getPreviousSibling();
  if (!maybeRecKeyword.isNull() && maybeRecKeyword.getType() == "rec") {
    return true;
  }
  return false;
}

} // namespace ocamlc2

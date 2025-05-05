#include <cassert>
#include <cpp-tree-sitter.h>
#include <iostream>
#include <ocamlc2/Parse/TSUtil.h>
#include <tree-sitter-ocaml.h>

using namespace ocamlc2;

int main() {
  auto ocaml = ::ts::Language{tree_sitter_ocaml()};
  auto parser = ::ts::Parser{ocaml};
  {
    const auto source = R"(
      let k ~(a:int) = a;;
    )";
    std::cout << source << '\n';
    auto tree = parser.parseString(source);
    auto node = tree.getRootNode();
    std::cout << node.getSExpr() << '\n';
    auto cursor = NamedCursor{node};
    while (cursor.getCurrentNode().getType() != "let_binding") {
      std::cout << cursor.getCurrentNode().getSExpr() << '\n';
      assert(cursor.gotoFirstNamedChild());
    }
    assert(cursor.gotoFirstNamedChild());
    assert(cursor.gotoNextNamedSibling());
    node = cursor.getCurrentNode();
    assert(node.getType() == "parameter");
    std::cout << node.getSExpr() << '\n';
    auto pattern = node.getChildByFieldName("pattern");
    assert(pattern.getType() == "value_pattern");
    std::cout << pattern.getSExpr() << '\n';
  }

  {
    const auto source = R"(
      let k ~a:a2 = a;;
    )";
    std::cout << source << '\n';
    auto tree = parser.parseString(source);
    auto node = tree.getRootNode();
    std::cout << node.getSExpr() << '\n';
    auto cursor = NamedCursor{node};
    while (cursor.getCurrentNode().getType() != "let_binding") {
      std::cout << cursor.getCurrentNode().getSExpr() << '\n';
      assert(cursor.gotoFirstNamedChild());
    }
    assert(cursor.gotoFirstNamedChild());
    assert(cursor.gotoNextNamedSibling());
    node = cursor.getCurrentNode();
    assert(node.getType() == "parameter");
    std::cout << node.getSExpr() << '\n';
    auto pattern = node.getChildByFieldName("pattern");
    assert(pattern.getType() == "value_pattern");
    std::cout << pattern.getSExpr() << '\n';
  }

  {
    const auto source = R"(
      let k a:int = a;;
    )";
    std::cout << source << '\n';
    auto tree = parser.parseString(source);
    auto node = tree.getRootNode();
    std::cout << node.getSExpr() << '\n';
    auto cursor = NamedCursor{node};
    while (cursor.getCurrentNode().getType() != "let_binding") {
      std::cout << cursor.getCurrentNode().getSExpr() << '\n';
      assert(cursor.gotoFirstNamedChild());
    }
    assert(cursor.gotoFirstNamedChild());
    assert(cursor.gotoNextNamedSibling());
    node = cursor.getCurrentNode();
    assert(node.getType() == "parameter");
    std::cout << node.getSExpr() << '\n';
    auto pattern = node.getChildByFieldName("pattern");
    assert(pattern.getType() == "value_pattern");
    std::cout << pattern.getSExpr() << '\n';
  }
  printf("pass\n");
  // CHECK: pass
  return 0;
}

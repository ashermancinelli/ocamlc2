import pytest
from fe import parse_ocaml, Tree


def test_parse_simple_expression():
    """Test parsing a simple OCaml expression."""
    source = "let x = 1 in x"
    tree = parse_ocaml(source)
    assert tree is not None
    assert not tree.root_node.has_error

    source2 = str(tree.root_node)
    assert source2 == "(compilation_unit (expression_item (let_expression (value_definition (let_binding pattern: (value_name) body: (number))) (value_path (value_name)))))"


def test_parse_invalid_expression():
    """Test parsing an invalid OCaml expression."""
    source = "let x = 1 in"  # Incomplete expression
    tree: Tree = parse_ocaml(source)
    assert tree is not None
    assert tree.root_node.has_error

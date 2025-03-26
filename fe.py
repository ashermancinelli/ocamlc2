import tree_sitter_ocaml as tsocaml
from tree_sitter import Language, Parser, Tree


def get_parser() -> Parser:
    lang = Language(tsocaml.language_ocaml())
    p = Parser(lang)
    return p


def parse_ocaml(src: str) -> Tree:
    p = get_parser()
    return p.parse(src.encode())

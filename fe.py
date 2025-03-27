from types import *
from typing import Optional

import tree_sitter_ocaml as tsocaml
from tree_sitter import Language, Node
from tree_sitter import Parser as TSParser
from tree_sitter import Tree


# ANSI escape codes for text formatting
class ANSIColors:
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Text styles
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

    # Reset
    RESET = "\033[0m"

    @staticmethod
    def colorize(text: str, color: str) -> str:
        return f"{color}{text}{ANSIColors.RESET}"

    @staticmethod
    def bold(text: str) -> str:
        return ANSIColors.colorize(text, ANSIColors.BOLD)

    @staticmethod
    def italic(text: str) -> str:
        return ANSIColors.colorize(text, ANSIColors.ITALIC)

    @staticmethod
    def underline(text: str) -> str:
        return ANSIColors.colorize(text, ANSIColors.UNDERLINE)


class Parser:
    def __init__(self):
        self.src: Optional[str] = None
        self.lang = Language(tsocaml.language_ocaml())
        self.p = TSParser(self.lang)
        self.parsed: Optional[Tree] = None

    def parse(self, src: str) -> Tree:
        self.src = src
        self.parsed = self.p.parse(src.encode())
        return self.parsed

    @staticmethod
    def byte_to_line_col(src: str, byte: int) -> tuple[int, int]:
        line = 1
        col = 0
        for i, c in enumerate(src):
            if i == byte:
                break
            if c == "\n":
                line += 1
                col = 0
            else:
                col += 1
        return line, col

    @staticmethod
    def pprint_kw(kw: str) -> str:
        significant_keywords = [
            "let_binding",
            "value_definition",
            "compilation_unit",
            "unit",
            "for_expression",
            "number",
            "do_clause",
            "value_pattern",
            "application_expression",
            "value_path",
            "string",
        ]
        if kw in significant_keywords:
            return ANSIColors.bold(ANSIColors.colorize(kw, ANSIColors.CYAN))
        return ANSIColors.italic(kw)

    @staticmethod
    def pprint_linecol(line: int, col: int) -> str:
        return ANSIColors.italic(f"{line}:{col}")

    def pprint(self, tree: Tree):
        def print_node(node: Node, indent: int = 0) -> None:
            print()
            typename = Parser.pprint_kw(node.type)
            assert self.src is not None
            line, col = Parser.byte_to_line_col(self.src, node.start_byte)
            linecol = Parser.pprint_linecol(line, col)
            print("  " * indent + f"({typename}:{linecol}", end="")
            if node.child_count > 0:
                for child in node.children:
                    print_node(child, indent + 1)

            print(")", end="")
            if indent == 0:
                print()  # Newline at the end of the tree

        print_node(tree.root_node)

    @staticmethod
    def _pprint(tree: Tree):
        s = str(tree.root_node)
        indent = 0
        for i, c in enumerate(s):
            # if c == ' ' and s[i - 1] != ':':
            if c == " ":
                print()
                print("  " * indent, end="")
                continue
            elif c == "(":
                indent += 1
            elif c == ")":
                indent -= 1
            print(c, end="")
        print()

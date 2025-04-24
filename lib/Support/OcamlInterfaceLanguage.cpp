#include "ocamlc2/Support/Utils.h"
#include <cpp-tree-sitter.h>
#include <tree-sitter-ocaml-interface.h>

ts::Language getOCamlInterfaceLanguage() {
  return ts::Language(tree_sitter_ocaml_interface());
}

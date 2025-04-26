#include "ocamlc2/Support/Utils.h"
#include <cpp-tree-sitter.h>
#include <tree-sitter-ocaml-interface.h>

namespace ocamlc2 {
ts::Language getOCamlInterfaceLanguage() {
  return ts::Language(tree_sitter_ocaml_interface());
}
} // namespace ocamlc2

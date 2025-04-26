
#include "ocamlc2/Support/Utils.h"
#include <cpp-tree-sitter.h>
#include <tree-sitter-ocaml.h>

namespace ocamlc2 {
ts::Language getOCamlLanguage() {
  return ts::Language(tree_sitter_ocaml());
}
} // namespace ocamlc2

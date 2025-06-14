#include "ocamlc2/Support/LLVMCommon.h"
#include <memory>

std::string getUniqueName(StringRef prefix) {
  static unsigned counter = 0;
  std::string name = prefix.str() + "$" + std::to_string(counter++);
  return name;
}

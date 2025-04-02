#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#define DEBUG_TYPE ""
#include "ocamlc2/Support/Debug.h.inc"

using namespace llvm;
static cl::opt<bool> debug("L", cl::desc("Enable debug mode"), cl::init(false));
static cl::opt<std::string> debugOnly("Lonly", cl::desc("Enable debug type"), cl::init(""));

namespace ocamlc2 {
  bool debug_enabled(const std::string &debug_type) {
    return debug || debugOnly == debug_type;
  }
}

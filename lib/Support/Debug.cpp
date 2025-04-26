#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "ocamlc2/Support/CL.h"
#define DEBUG_TYPE ""
#include "ocamlc2/Support/Debug.h.inc"

using namespace llvm;
static cl::opt<std::string> debugOnly("Lonly", cl::desc("Enable debug type"), cl::init(""));

namespace ocamlc2 {
  bool debug_enabled(const std::string &debug_type) {
    return CL::Debug || debugOnly == debug_type;
  }
}

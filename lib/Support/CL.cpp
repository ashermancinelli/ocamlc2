#include "ocamlc2/Support/CL.h"
#include <unistd.h>

using namespace llvm;

namespace ocamlc2::CL {
bool Debug = false;
bool RunGDB = false;
bool Color = true;
bool DumpTypes = false;
bool Freestanding = false;
bool StdlibOnly = false;
bool DParseTree = false;
bool DTypedTree = false;
llvm::cl::OptionCategory OcamlOptions("OCaml Options", "");
}
using namespace ocamlc2::CL;

static cl::opt<bool, true> runGdb("gdb", cl::desc("Run the program under gdb"),
                                  cl::location(RunGDB), cl::cat(OcamlOptions));

static cl::opt<std::string> debugger("debugger",
                                     cl::desc("The debugger to use"),
                                     cl::init("lldb"), cl::cat(OcamlOptions));

static cl::opt<bool, true> debug("L", cl::desc("Enable debug mode"),
                                 cl::location(Debug), cl::cat(OcamlOptions));

static cl::opt<bool, true> dumpTypes("dtypes", cl::desc("Dump types"),
                                     cl::location(DumpTypes),
                                     cl::cat(OcamlOptions));

static cl::alias DdumpTypes("d", cl::desc("Dump types"),
                            cl::aliasopt(dumpTypes), cl::cat(OcamlOptions));

static cl::opt<bool>
    noColor("no-color", cl::desc("Disable color output"),
            cl::cb<void, bool>([](bool value) { Color = !value; }),
            cl::cat(OcamlOptions));

static cl::opt<bool>
    freestanding("freestanding", cl::desc("Enable freestanding mode"),
                 cl::cb<void, bool>([](bool value) { Freestanding = value; }),
                 cl::cat(OcamlOptions));
static cl::alias Ffreestanding("f", cl::desc("Enable freestanding mode"),
                              cl::aliasopt(freestanding), cl::cat(OcamlOptions));

static cl::opt<bool, true> clStdlibOnly("fstdlib-only",
                                  cl::desc("Enable stdlib-only mode"),
                                  cl::location(StdlibOnly),
                                  cl::cat(OcamlOptions));

static cl::opt<bool, true> clDParseTree("dparsetree",
                                  cl::desc("Enable parse tree dump"),
                                  cl::location(DParseTree),
                                  cl::cat(OcamlOptions));

static cl::opt<bool, true> clDTypedTree("dtypedtree",
                                  cl::desc("Enable typed tree dump"),
                                  cl::location(DTypedTree),
                                  cl::cat(OcamlOptions));

void maybeReplaceWithGDB(int argc, char **argv) {
  if (!RunGDB) {
    return;
  }
  std::vector<char *> newArgs;
  newArgs.push_back(const_cast<char *>(debugger.getValue().c_str()));
  newArgs.push_back(const_cast<char *>("--"));
  newArgs.push_back(argv[0]);
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-gdb" or arg == "--gdb") {
      continue;
    }
    newArgs.push_back(argv[i]);
  }
  newArgs.push_back(nullptr);
  auto debuggerExe = debugger.getValue().c_str();
  execvp(debuggerExe, newArgs.data());
}

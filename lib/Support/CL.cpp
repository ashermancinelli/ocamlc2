#include "ocamlc2/Support/CL.h"
#include <llvm/Support/CommandLine.h>
#include <unistd.h>

using namespace llvm;

bool Debug = false;
bool RunGDB = false;
bool Color = true;
bool DumpTypes = false;
static cl::opt<bool, true> runGdb("gdb", cl::desc("Run the program under gdb"),
                                  cl::location(RunGDB));

static cl::opt<std::string> debugger("debugger", cl::desc("The debugger to use"), cl::init("lldb"));

static cl::opt<bool, true> debug("L", cl::desc("Enable debug mode"),
                           cl::location(Debug));

static cl::opt<bool, true> dumpTypes("dump-types", cl::desc("Dump types"),
                                     cl::location(DumpTypes));

static cl::opt<bool>
    noColor("no-color", cl::desc("Disable color output"),
            cl::cb<void, bool>([](bool value) { Color = !value; }));

void maybeReplaceWithGDB(int argc, char **argv) {
  if (!RunGDB) {
    return;
  }
  std::vector<char*> newArgs;
  newArgs.push_back(const_cast<char*>(debugger.getValue().c_str()));
  newArgs.push_back(const_cast<char*>("--"));
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

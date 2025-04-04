#include "ocamlc2/Support/CL.h"
#include <llvm/Support/CommandLine.h>

using namespace llvm;

bool Debug = false;
bool RunGDB = false;

static cl::opt<bool, true> runGdb("gdb", cl::desc("Run the program under gdb"),
                                  cl::location(RunGDB));

static cl::opt<bool, true> debug("L", cl::desc("Enable debug mode"),
                           cl::location(Debug));

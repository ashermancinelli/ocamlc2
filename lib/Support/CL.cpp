#include "ocamlc2/Support/CL.h"
#include <unistd.h>
#include "ocamlc2/OCamlC2Config.h"
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <filesystem>

using namespace llvm;

namespace ocamlc2::CL {
namespace {
static std::string debugger = "lldb";
}
namespace fs = std::filesystem;
bool Debug = false;
bool RunGDB = false;
bool Color = true; // llvm::WithColor::defaultAutoDetectFunction()(llvm::outs());
bool DumpTypes = false;
bool DumpStdlib = false;
bool Freestanding = false;
bool StdlibOnly = false;
bool DParseTree = false;
bool DTypedTree = false;
bool Repl = false;
bool InRLWrap = false;
bool Quiet = true;
bool PreprocessWithCPPO = false;
bool OCamlFormat = true;

llvm::cl::OptionCategory OcamlOptions("OCaml Options", "");

void maybeReplaceWithGDB(int argc, char **argv) {
  if (!RunGDB) {
    return;
  }
  std::vector<char *> newArgs;
  newArgs.push_back(const_cast<char *>(debugger.c_str()));
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
  auto debuggerExe = debugger.c_str();
  execvp(debuggerExe, newArgs.data());
}
}

using namespace ocamlc2::CL;

static cl::opt<bool, true> preprocessWithCPPO("cpp",
                                             cl::desc("Preprocess with CPPO"),
                                             cl::location(PreprocessWithCPPO),
                                             cl::cat(OcamlOptions));

static cl::opt<bool, true> quiet("quiet", cl::desc("Disable AST and type printing after each command"),
                                 cl::location(Quiet), cl::cat(OcamlOptions));

static cl::alias Quiet("q", cl::desc("Disable AST and type printing after each command"),
                       cl::aliasopt(quiet), cl::cat(OcamlOptions));

static cl::opt<bool, true> repl("repl", cl::desc("Run the REPL"),
                               cl::location(Repl), cl::cat(OcamlOptions));
static cl::alias Repl("r", cl::desc("Run the REPL"),
                      cl::aliasopt(repl), cl::cat(OcamlOptions));

static cl::opt<bool, true>
    inRLWrap("in-rlwrap",
             cl::desc("Internal option meaning the executable is already "
                      "running itself under rlwrap"),
             cl::location(InRLWrap), cl::cat(OcamlOptions));

static cl::opt<bool, true> runGdb("gdb", cl::desc("Run the program under gdb"),
                                  cl::location(RunGDB), cl::cat(OcamlOptions));

static cl::opt<std::string, true> clDebugger("debugger",
                                     cl::desc("The debugger to use"),
                                     cl::location(debugger), cl::cat(OcamlOptions));

static cl::opt<bool, true> debug("L", cl::desc("Enable debug mode"),
                                 cl::location(Debug), cl::cat(OcamlOptions));

static cl::opt<bool, true> dumpTypes("dtypes", cl::desc("Dump types"),
                                     cl::location(DumpTypes),
                                     cl::cat(OcamlOptions));

static cl::opt<bool, true> dumpStdlib("dstdlib", cl::desc("Dump stdlib"),
                                     cl::location(DumpStdlib),
                                     cl::cat(OcamlOptions));

static cl::alias DdumpTypes("d", cl::desc("Dump types"),
                            cl::aliasopt(dumpTypes), cl::cat(OcamlOptions));

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

static cl::opt<bool> clNoColor("no-color",
                                  cl::desc("Disable color output"),
                                  cl::cb<void, bool>([](bool value) { Color = !value; }),
                                  cl::cat(OcamlOptions));

static cl::opt<bool, true> clOCamlFormat("fmt",
                                  cl::desc("Enable ocamlformat"),
                                  cl::location(OCamlFormat),
                                  cl::cat(OcamlOptions));

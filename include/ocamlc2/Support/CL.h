#pragma once
#include <llvm/Support/CommandLine.h>
namespace ocamlc2::CL {
extern bool Debug, RunGDB, Color, DumpTypes, DumpStdlib, Freestanding, StdlibOnly,
    DParseTree, DTypedTree, Repl, InRLWrap, Quiet, PreprocessWithCPPO, OCamlFormat;
void maybeReplaceWithGDB(int argc, char **argv);
extern llvm::cl::OptionCategory OcamlOptions;
}

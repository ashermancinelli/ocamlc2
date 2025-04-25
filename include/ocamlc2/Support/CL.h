#pragma once
#include <llvm/Support/CommandLine.h>
extern bool Debug, RunGDB, Color, DumpTypes, Freestanding;
void maybeReplaceWithGDB(int argc, char **argv);
extern llvm::cl::OptionCategory OcamlOptions;

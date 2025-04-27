#include "ocamlc2/Support/Repl.h"
#include "ocamlc2/Support/CL.h"
#include <iostream>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Program.h>
#include <unistd.h>

#define DEBUG_TYPE "repl"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

static fs::path findRlwrap() {
  llvm::StringRef path = getenv("PATH");
  if (path.empty()) {
    return {};
  }
  while (true) {
    auto [currPath, next] = path.split(llvm::sys::EnvPathSeparator);
    if (auto rlwrap = fs::path(currPath.str()) / "rlwrap"; fs::exists(rlwrap)) {
      return rlwrap;
    }
    if (next.empty()) {
      break;
    }
    path = next;
  }
  return {};
}

static void runUnderRlwrap(int argc, char **argv, fs::path exe, Unifier &unifier) {
  auto rlwrap = findRlwrap();
  if (rlwrap.empty()) {
    return;
  }
  SmallVector<std::string> newArgs = {rlwrap, exe};
  for (int i = 1; i < argc; i++) {
    newArgs.emplace_back(argv[i]);
  }
  newArgs.push_back("--in-rlwrap");
  SmallVector<char *> newArgsC = llvm::map_to_vector(newArgs, [](std::string &s) { return s.data(); });
  DBGS("Running under rlwrap: " << llvm::join(newArgs, " "));
  execvp(rlwrap.c_str(), newArgsC.data());
}

[[noreturn]] void runRepl(int argc, char **argv, fs::path exe, Unifier &unifier) {
  if (not CL::InRLWrap) {
    runUnderRlwrap(argc, argv, exe, unifier);
  }
  std::string line;
  std::cout << "OCaml REPL (type '.' on a line by itself to finish)\n";
  while (true) {
    std::string source = "";
    while (std::getline(std::cin, line)) {
      if (line == ".") {
        break;
      }
      source += line;
    }
    if (source.empty()) {
      std::exit(unifier.anyFatalErrors());
    }
    unifier.loadSource(source);
    if (failed(unifier)) {
      unifier.showErrors();
    } else {
      unifier.showTypedTree();
    }
  }
}
} // namespace ocamlc2

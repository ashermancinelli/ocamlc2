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
  DBGS("Running under rlwrap: " << llvm::join(newArgs, " ") << '\n');
  execvp(rlwrap.c_str(), newArgsC.data());
}

static void typeOf(Unifier &unifier, ArrayRef<std::string> args) {
  if (args.size() != 1) {
    llvm::errs() << "Usage: #type <symbol>\n";
    return;
  }
  unifier.showType(llvm::outs() << "val ", args[0]) << "\n";
}

static void showAST(Unifier &unifier, ArrayRef<std::string> args) {
  unifier.showParseTree();
}

static void showTypedAST(Unifier &unifier, ArrayRef<std::string> args) {
  unifier.showTypedTree();
}

[[noreturn]] void runRepl(int argc, char **argv, fs::path exe, Unifier &unifier) {
  static std::unordered_map<std::string, std::function<void(Unifier &, ArrayRef<std::string>)>> commands = {
    {"type", typeOf},
    {"parsetree", showAST},
    {"typedtree", showTypedAST},
  };
  if (not CL::InRLWrap) {
    runUnderRlwrap(argc, argv, exe, unifier);
  }
  unifier.setMaxErrors(1);
  std::string line;
  std::cout << "OCaml REPL (type '.' on a line by itself to finish)\n";
  while (true) {
    std::string source = "";
    while (std::getline(std::cin, line)) {
      if (line == ".") {
        break;
      }
      if (line.starts_with("#")) {
        auto command = line.substr(1);
        auto args = llvm::split(command, " ");
        auto cmd = args.begin();
        auto it = commands.find((*cmd).str());
        if (it == commands.end()) {
          llvm::errs() << "Unknown command: " << *cmd << '\n';
          continue;
        }
        SmallVector<std::string> rest(args.begin(), args.end());
        rest.erase(rest.begin());
        it->second(unifier, rest);
        source = "";
        continue;
      }
      source += line;
    }
    if (source.empty()) {
      continue;
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

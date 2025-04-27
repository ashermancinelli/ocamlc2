#include "ocamlc2/Support/Repl.h"
#include "ocamlc2/Parse/TSUnifier.h"
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

struct Command {
  virtual void callback(Unifier &unifier, ArrayRef<std::string> args) = 0;
  virtual std::string_view help() const = 0;
  virtual std::string_view name() const = 0;
  virtual ~Command() = default;
};

struct TypeCommand : public Command {
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    if (args.size() != 1) {
      llvm::errs() << "Usage: #type <symbol>\n";
      return;
    }
    unifier.showType(llvm::outs() << "val ", args[0]) << "\n";
  }
  std::string_view help() const override {
    return "Show the type of a symbol";
  }
  std::string_view name() const override {
    return "type";
  }
};

struct ShowASTCommand : public Command {
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    unifier.showParseTree();
  }
  std::string_view help() const override {
    return "Show the parse tree";
  }
  std::string_view name() const override {
    return "parsetree";
  }
};

struct ShowTypedASTCommand : public Command {
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    unifier.showTypedTree();
  }
  std::string_view help() const override {
    return "Show the typed tree";
  }
  std::string_view name() const override {
    return "typedtree";
  }
};

struct ShowSourceCommand : public Command {
  std::string &sourceSoFar;
  ShowSourceCommand(std::string &sourceSoFar) : sourceSoFar(sourceSoFar) {}
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    llvm::outs() << sourceSoFar << '\n';
  }
  std::string_view help() const override {
    return "Show the code typed into the repl so far";
  }
  std::string_view name() const override {
    return "source";
  }
};

struct HelpCommand : public Command {
  HelpCommand(std::vector<std::unique_ptr<Command>> &commands) : commands(commands) {}
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    llvm::outs() << "Available commands:\n";
    for (const auto &cmd : commands) {
      llvm::outs() << "  " << cmd->name() << ": " << cmd->help() << '\n';
    }
  }
  std::string_view help() const override {
    return "Show available commands";
  }
  std::string_view name() const override {
    return "help";
  }
private:
  std::vector<std::unique_ptr<Command>> &commands;
};

struct QuietCommand : public Command {
  bool &quiet;
  QuietCommand(bool &quiet) : quiet(quiet) {}
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    quiet = true;
  }
  std::string_view help() const override {
    return "Disable AST and type printing after each command";
  }
  std::string_view name() const override {
    return "quiet";
  }
};

[[noreturn]] void exitRepl(Unifier &unifier) {
  llvm::outs() << "Goodbye!\n";
  std::exit(unifier.anyFatalErrors());
}

[[noreturn]] void runRepl(int argc, char **argv, fs::path exe, Unifier &unifier) {
  std::string sourceSoFar;
  static std::vector<std::unique_ptr<Command>> commands;
  commands.emplace_back(std::make_unique<TypeCommand>());
  commands.emplace_back(std::make_unique<ShowASTCommand>());
  commands.emplace_back(std::make_unique<ShowTypedASTCommand>());
  commands.emplace_back(std::make_unique<ShowSourceCommand>(sourceSoFar));
  commands.emplace_back(std::make_unique<HelpCommand>(commands));
  commands.emplace_back(std::make_unique<QuietCommand>(CL::Quiet));

  if (not CL::InRLWrap) {
    runUnderRlwrap(argc, argv, exe, unifier);
  }
  unifier.setMaxErrors(1);
  std::string line;
  std::cout << "OCaml REPL (type '.' on a line by itself to finish the command, or #help for help)\n";
  while (true) {
    std::string source = "";
    llvm::outs() << "> ";
    while (std::getline(std::cin, line)) {
      TRACE();
      if (!std::cin) {
        TRACE();
        exitRepl(unifier);
      }
      if (line == "") {
        TRACE();
        break;
      }
      if (line.starts_with("#")) {
        TRACE();
        auto command = line.substr(1);
        auto args = llvm::split(command, " ");
        auto cmd = args.begin();
        auto it = llvm::find_if(commands, [&](const auto &c) {
          return std::string_view(*cmd) == c->name();
        });
        if (it == commands.end()) {
          llvm::errs() << "Unknown command: " << *cmd << '\n';
          continue;
        }
        sourceSoFar += "(* " + line + " *)\n";
        SmallVector<std::string> rest(args.begin(), args.end());
        rest.erase(rest.begin());
        (*it)->callback(unifier, rest);
        source = "";
        llvm::outs() << "> ";
        continue;
      }
      TRACE();
      sourceSoFar += line + '\n';
      source += line;
    }
    if (source.empty()) {
      if (std::cin.eof() or std::cin.fail() or std::cin.bad()) {
        exitRepl(unifier);
      }
      continue;
    }
    unifier.loadSource(source);
    if (failed(unifier)) {
      unifier.showErrors();
    } else if (not CL::Quiet) {
      unifier.showTypedTree();
    }
  }
}
} // namespace ocamlc2

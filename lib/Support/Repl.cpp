#include "ocamlc2/Support/Repl.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Support/Colors.h"
#include <iostream>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/Process.h>
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
  SmallVector<std::string> newArgs = {rlwrap, "--no-warnings",
                                      "--complete-filenames", exe};
  for (int i = 1; i < argc; i++) {
    newArgs.emplace_back(argv[i]);
  }
  newArgs.push_back("--in-rlwrap");
  DBGS("Running under rlwrap: " << llvm::join(newArgs, " ") << '\n');
  SmallVector<char *> newArgsC = llvm::map_to_vector(newArgs, [](std::string &s) { return s.data(); });
  newArgsC.push_back(nullptr);
  execvp(rlwrap.c_str(), newArgsC.data());
  llvm_unreachable("Should not return");
}

struct Command {
  virtual void callback(Unifier &unifier, ArrayRef<std::string> args) = 0;

  llvm::raw_ostream& help(llvm::raw_ostream &os) const { return os << help(); }
  llvm::raw_ostream& name(llvm::raw_ostream &os) const { return os << name(); }

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
      cmd->name(llvm::outs() << "  #" << ANSIColors::bold())
          << ANSIColors::reset() << ": ";
      cmd->help(llvm::outs()) << '\n';
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

struct ShellCommand : public Command {
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    if (args.size() != 1) {
      llvm::errs() << "Usage: #sh <command>\n";
      return;
    }
    SmallVector<StringRef> newArgs = llvm::map_to_vector(args, [](auto &s) { return StringRef(s); });
    auto path = llvm::sys::findProgramByName(StringRef(args[0]));
    if (not path) {
      llvm::errs() << "Command not found: " << args[0] << '\n';
      return;
    }
    DBGS("Running command: " << *path << " " << llvm::join(newArgs, " ") << '\n');
    int rc = llvm::sys::ExecuteAndWait(*path, newArgs, {}, {}, {});
    if (rc != 0) {
      llvm::errs() << "Command failed with status " << rc << '\n';
    }
  }
  std::string_view help() const override {
    return "Run a shell command";
  }
  std::string_view name() const override {
    return "sh";
  }
};

struct ClearCommand : public Command {
  std::string &sourceSoFar;
  ClearCommand(std::string &sourceSoFar) : sourceSoFar(sourceSoFar) {}
  void callback(Unifier &unifier, ArrayRef<std::string> args) override {
    sourceSoFar = "";
  }
  std::string_view help() const override {
    return "Clear the current module";
  }
  std::string_view name() const override {
    return "clear";
  }
};

[[noreturn]] void exitRepl(Unifier &unifier) {
  llvm::outs() << ANSIColors::faint() << ANSIColors::italic() << "Goodbye!\n" << ANSIColors::reset();
  std::exit(unifier.anyFatalErrors());
}

[[noreturn]] void runRepl(int argc, char **argv, fs::path exe, Unifier &unifier) {
  if (!llvm::sys::Process::StandardInIsUserInput()) {
    llvm::errs() << "Error: Standard input is not a tty, can't start REPL\n";
    std::exit(1);
  }
  std::string sourceSoFar;
  static std::vector<std::unique_ptr<Command>> commands;
  commands.emplace_back(std::make_unique<TypeCommand>());
  commands.emplace_back(std::make_unique<ShowASTCommand>());
  commands.emplace_back(std::make_unique<ShowTypedASTCommand>());
  commands.emplace_back(std::make_unique<ShowSourceCommand>(sourceSoFar));
  commands.emplace_back(std::make_unique<HelpCommand>(commands));
  commands.emplace_back(std::make_unique<QuietCommand>(CL::Quiet));
  commands.emplace_back(std::make_unique<ShellCommand>());
  commands.emplace_back(std::make_unique<ClearCommand>(sourceSoFar));
  if (not CL::InRLWrap) {
    runUnderRlwrap(argc, argv, exe, unifier);
    assert(false && "Should not return");
  }
  unifier.setMaxErrors(1);
  std::string line;
  std::cout << ANSIColors::bold() << "OCaml REPL" << ANSIColors::reset() << "\n"
            << ANSIColors::faint() << ANSIColors::italic()
            << "(type an empty line by itself to finish the command, or #help "
               "for more commands)\n"
            << ANSIColors::reset();
  auto newPrompt = [&] {
    llvm::outs() << ANSIColors::faint() << "> " << ANSIColors::reset();
  };
  auto continuePrompt = [&] {
    llvm::outs() << ANSIColors::faint() << ". " << ANSIColors::reset();
  };
  while (true) {
    std::string source = "";
    newPrompt();
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
        newPrompt();
        continue;
      }
      TRACE();
      sourceSoFar += line + '\n';
      source += line;
      continuePrompt();
    }
    if (source.empty()) {
      if (std::cin.eof() or std::cin.fail() or std::cin.bad()) {
        exitRepl(unifier);
      }
      continue;
    }
    unifier.loadSource(sourceSoFar);
    if (failed(unifier)) {
      unifier.showErrors();
      sourceSoFar = "";
    } else if (not CL::Quiet) {
      unifier.showTypedTree();
    }
  }
}
} // namespace ocamlc2

#include "llvm/Support/Process.h"
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <cstdint>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/iterator.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <llvm/Support/FileSystem.h>
#include <string>
#include <cpp-subprocess/subprocess.hpp>

#define DEBUG_TYPE "sourceutils"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"

namespace ocamlc2 {

llvm::raw_ostream& Unifier::show(bool showUnnamed, bool showTypes) {
  return show(sources.back().tree.getRootNode().getCursor(), showUnnamed, showTypes);
}
llvm::raw_ostream& Unifier::showParseTree() {
  return show(true, false);
}
llvm::raw_ostream& Unifier::showTypedTree() {
  return show(false, true);
}

llvm::raw_ostream& Unifier::show(ts::Cursor cursor, bool showUnnamed, bool showTypes) {
  auto showTypesCallback = [this](llvm::raw_ostream &os, ts::Node node) {
    if (auto *te = nodeToType.lookup(node.getID())) {
      os << ANSIColors::magenta() << " " << *te << ANSIColors::reset();
    }
  };
  auto callback = showTypes ? std::optional{showTypesCallback} : std::nullopt;
  return dump(llvm::errs(), cursor.copy(), sources.back().source, 0, showUnnamed, callback);
}

TypeExpr* Unifier::infer(ts::Node const& ast) {
  return infer(ast.getCursor());
}

Unifier::Unifier() {
  if (failed(initializeEnvironment())) {
    llvm::errs() << "Failed to initialize environment\n";
    exit(1);
  }
}

Unifier::Unifier(std::string filepath) {
  TRACE();
  if (failed(initializeEnvironment())) {
    llvm::errs() << "Failed to initialize environment\n";
    exit(1);
  }
  loadSourceFile(filepath);
}

void Unifier::loadSource(llvm::StringRef source) {
  DBGS("Source:\n" << source << "\n");
  ::ts::Language language = getOCamlLanguage();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  static const std::string replName = "repl.ml";
  sources.emplace_back(replName, source.str(), std::move(tree));

  // Make sure the symbols defined in the repl are always visible.
  auto moduleName = stringArena.save(filePathToModuleName(replName));
  pushModuleSearchPath(moduleName);

  (void)infer(sources.back().tree.getRootNode().getCursor());
}

void Unifier::loadSourceFile(fs::path filepath) {
  TRACE();
  if (CL::PreprocessWithCPPO) {
    auto preprocessed = preprocessWithCPPO(filepath);
    if (failed(preprocessed)) {
      llvm::errs() << "Failed to preprocess file: " << filepath << '\n';
      return;
    }
    filepath = preprocessed.value();
  }
  if (filepath == "-") {
    auto source = slurpStdin();
    if (failed(source)) {
      llvm::errs() << "Failed to read from stdin\n";
      return;
    }
    loadSource(source.value());
  } else if (filepath.extension() == ".ml") {
    loadImplementationFile(filepath);
  } else if (filepath.extension() == ".mli") {
    loadInterfaceFile(filepath);
  } else {
    llvm::errs() << "Unknown file extension: " << filepath << '\n';
    assert(false && "Unknown file extension");
  }
  (void)infer(sources.back().tree.getRootNode().getCursor());
}

void Unifier::loadImplementationFile(fs::path filepath) {
  DBGS("Loading implementation file: " << filepath << "\n");
  std::string source = must(slurpFile(filepath));
  if (source.empty()) {
    DBGS("Source is empty\n");
    return;
  }
  DBGS("Source:\n" << source << "\n");
  ::ts::Language language = getOCamlLanguage();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  sources.emplace_back(filepath, source, std::move(tree));
}

void Unifier::loadInterfaceFile(fs::path filepath) {
  DBGS("Loading interface file: " << filepath << "\n");
  std::string source = must(slurpFile(filepath));
  if (source.empty()) {
    DBGS("Source is empty\n");
    return;
  }
  DBGS("Source:\n" << source << "\n");
  ::ts::Language language = getOCamlInterfaceLanguage();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  sources.emplace_back(filepath, source, std::move(tree));
}

llvm::StringRef Unifier::getTextSaved(Node node) {
  auto sv = ocamlc2::getText(node, sources.back().source);
  return stringArena.save(sv);
}

void Unifier::setMaxErrors(int maxErrors) {
  TRACE();
  this->maxErrors = maxErrors < 0 ? 100 : maxErrors;
}

void Unifier::loadStdlibInterfaces(fs::path exe) {
  DBGS("Disabling debug while loading stdlib interfaces\n");
  isLoadingStdlib = true;
  bool savedDebug = CL::Debug;
  CL::Debug = false;
  for (auto filepath : getStdlibOCamlInterfaceFiles(exe)) {
    DBGS("Loading stdlib interface file: " << filepath << "\n");
    loadSourceFile(filepath);
  }
  isLoadingStdlib = false;
  CL::Debug = savedDebug;
  DBGS("Done loading stdlib interfaces\n");
}

llvm::raw_ostream& operator<<(llvm::raw_ostream &os, subprocess::Buffer const& buf) {
  for (auto c : buf.buf) {
    os << c;
  }
  return os;
}

static void runOCamlFormat(ModuleOperator *module, llvm::raw_ostream &os) {
  auto ocamlFormat = llvm::sys::findProgramByName("ocamlformat");
  auto bat = llvm::sys::findProgramByName("bat");
  if (!ocamlFormat || !bat) {
    llvm::errs() << "NOTE: Failed to find ocamlformat or bat\n";
    module->decl(os) << '\n';
    return;
  }

  fs::path tmpDir = fs::temp_directory_path();
  std::string randomName =
      "m" + std::to_string(std::random_device{}()) + std::string(".mli");
  std::string tmpFileName = (tmpDir / randomName).string();
  std::error_code ec;
  llvm::raw_fd_ostream fs(tmpFileName, ec);
  if (ec) {
    llvm::errs() << "Failed to create tmp file: " << ec.message() << '\n';
    module->decl(llvm::outs()) << '\n';
  }

  module->decl(fs) << '\n';
  fs.close();

  std::vector<std::string> args = {*ocamlFormat, "--enable-outside-detected-project", "--intf", tmpFileName, "-i"};
  DBGS("Running ocamlformat with args:\n" << llvm::join(args, " ") << '\n');
  auto formatProcess = subprocess::check_output(args);

  if (!llvm::sys::Process::StandardOutIsDisplayed() or !CL::Color) {
    os << subprocess::check_output({"cat", tmpFileName});
    return;
  }

  std::vector<std::string> highlightArgs = {*bat, "-lml", "--plain", "--color", "always", tmpFileName};
  DBGS("Running bat with args:\n" << llvm::join(highlightArgs, " ") << '\n');
  os << subprocess::check_output(highlightArgs);
}

void Unifier::dumpTypes(llvm::raw_ostream &os, bool showStdlib) {
  if (!diagnostics.empty()) {
    return showErrors();
  }
  SignatureOperator::useNewline('\n');
  auto show = [&](ModuleOperator *module) {
    if (CL::OCamlFormat) {
      runOCamlFormat(module, os);
    } else {
      module->decl(os) << '\n';
    }
  };
  if (showStdlib) {
    for (auto module : stdlibModules) {
      show(module);
    }
  }
  for (auto module : modulesToDump) {
    show(module);
  }
}

bool Unifier::anyFatalErrors() const { return !diagnostics.empty(); }
void Unifier::showErrors() {
  for (auto diag : diagnostics) {
    llvm::errs() << diag << "\n";
  }
}

static void stringToPath(llvm::StringRef name, llvm::SmallVector<llvm::StringRef> &pathSoFar) {
  auto [head, tail] = name.split('.');
  pathSoFar.push_back(head);
  while (!tail.empty()) {
    std::tie(head, tail) = tail.split('.');
    pathSoFar.push_back(head);
  }
}

static SmallVector<llvm::StringRef> stringToPath(llvm::StringRef name) {
  SmallVector<llvm::StringRef> path;
  stringToPath(name, path);
  return path;
}

llvm::raw_ostream &Unifier::showType(llvm::raw_ostream &os, llvm::StringRef name) {
  auto path = stringToPath(name);
  if (auto *type = maybeGetVariableType(path)) {
    os << name << " : " << *type;
  } else {
    os << "Type unknown for symbol: '" << name << "'\n";
  }
  return os;
}

detail::Scope::Scope(Unifier *unifier)
    : unifier(unifier), envScope(unifier->env()),
      concreteTypes(unifier->concreteTypes) {
  DBGS("open scope with concrete:\n");
  for (auto *type : unifier->concreteTypes) {
    DBGS("  " << *type << '\n');
  }
}

detail::Scope::~Scope() {
  DBGS("close scope with concrete:\n");
  unifier->concreteTypes = std::move(concreteTypes);
  for (auto *type : unifier->concreteTypes) {
    DBGS("  " << *type << '\n');
  }
}

Unifier::TypeVarEnvScope::TypeVarEnvScope(Unifier::TypeVarEnv &env) : scope(env) {
  DBGS("open type var env scope\n");
}

Unifier::TypeVarEnvScope::~TypeVarEnvScope() {
  DBGS("close type var env scope\n");
}

static unsigned openCount = 0;
void detail::ConcreteTypeVariableScope::logOpen() {
  DBGS("ConcreteTypeVariableScope::open with open scope count: " << openCount << " and " << concreteTypes.size() << " concrete types\n");
  ++openCount;
}

void detail::ConcreteTypeVariableScope::logClose() {
  --openCount;
  DBGS("ConcreteTypeVariableScope::close now with open scope count: " << openCount << " and " << concreteTypes.size() << " concrete types\n");
}

} // namespace ocamlc2

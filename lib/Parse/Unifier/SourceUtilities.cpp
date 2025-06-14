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

#define DEBUG_TYPE "SourceUtilities.cpp"
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
    showErrors();
    llvm::errs() << "Failed to initialize environment\n";
    exit(1);
  }
  if (failed(loadSourceFile(filepath))) {
    showErrors();
    llvm::errs() << "Failed to load source file\n";
    exit(1);
  }
}

LogicalResult Unifier::loadSource(llvm::StringRef source) {
  DBGS("Source:\n" << source << "\n");
  ::ts::Language language = getOCamlLanguage();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  if (failed(checkForSyntaxErrors(tree.getRootNode()))) {
    llvm::errs() << "Syntax errors found in source\n";
    return failure();
  }
  static const std::string replName = "repl.ml";
  sources.emplace_back(replName, source.str(), std::move(tree));

  // Make sure the symbols defined in the repl are always visible.
  auto moduleName = stringArena.save(filePathToModuleName(replName));
  pushModuleSearchPath(moduleName);

  (void)infer(sources.back().tree.getRootNode().getCursor());
  return success();
}

LogicalResult Unifier::loadSourceFile(fs::path filepath) {
  TRACE();
  if (CL::PreprocessWithCPPO) {
    auto preprocessed = preprocessWithCPPO(filepath);
    if (failed(preprocessed)) {
      llvm::errs() << "Failed to preprocess file: " << filepath << '\n';
      return failure();
    }
    filepath = preprocessed.value();
  }
  if (filepath == "-") {
    auto source = slurpStdin();
    if (failed(source)) {
      llvm::errs() << "Failed to read from stdin\n";
      return failure();
    }
    if (failed(loadSource(source.value()))) {
      llvm::errs() << "Failed to load source from stdin\n";
      return failure();
    }
  } else if (filepath.extension() == ".ml") {
    if (failed(loadImplementationFile(filepath))) {
      llvm::errs() << "Failed to load implementation file\n";
      return failure();
    }
  } else if (filepath.extension() == ".mli") {
    if (failed(loadInterfaceFile(filepath))) {
      llvm::errs() << "Failed to load interface file\n";
      return failure();
    }
  } else {
    llvm::errs() << "Unknown file extension: " << filepath << '\n';
    assert(false && "Unknown file extension");
  }
  (void)infer(sources.back().tree.getRootNode().getCursor());
  return success(not anyFatalErrors());
}

LogicalResult Unifier::loadImplementationFile(fs::path filepath) {
  DBGS("Loading implementation file: " << filepath << "\n");
  std::string source = must(slurpFile(filepath));
  if (source.empty()) {
    DBGS("Source is empty\n");
    return failure();
  }
  DBGS("Source:\n" << source << "\n");
  ::ts::Language language = getOCamlLanguage();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  if (failed(checkForSyntaxErrors(tree.getRootNode()))) {
    llvm::errs() << "Syntax errors found in source\n";
    return failure();
  }
  sources.emplace_back(filepath, source, std::move(tree));
  return success(not anyFatalErrors());
}

LogicalResult Unifier::loadInterfaceFile(fs::path filepath) {
  DBGS("Loading interface file: " << filepath << "\n");
  std::string source = must(slurpFile(filepath));
  if (source.empty()) {
    DBGS("Source is empty\n");
    return failure();
  }
  DBGS("Source:\n" << source << "\n");
  ::ts::Language language = getOCamlInterfaceLanguage();
  ::ts::Parser parser{language};
  auto tree = parser.parseString(source);
  if (failed(checkForSyntaxErrors(tree.getRootNode()))) {
    llvm::errs() << "Syntax errors found in source\n";
    return failure();
  }
  sources.emplace_back(filepath, source, std::move(tree));
  return success(not anyFatalErrors());
}

llvm::StringRef Unifier::getTextSaved(Node node) {
  auto sv = ocamlc2::getText(node, sources.back().source);
  return stringArena.save(sv);
}

void Unifier::setMaxErrors(int maxErrors) {
  TRACE();
  this->maxErrors = maxErrors < 0 ? 100 : maxErrors;
}

LogicalResult Unifier::loadStdlibInterfaces(fs::path exe) {
  DBGS("Disabling debug while loading stdlib interfaces\n");
  isLoadingStdlib = true;
  bool savedDebug = CL::Debug;
  CL::Debug = CL::DumpStdlib;
  for (auto filepath : getStdlibOCamlInterfaceFiles(exe)) {
    DBGS("Loading stdlib interface file: " << filepath << "\n");
    if (failed(loadSourceFile(filepath))) {
      llvm::errs() << "Failed to load stdlib interface file\n";
      return failure();
    }
  }
  isLoadingStdlib = false;
  CL::Debug = savedDebug;
  DBGS("Done loading stdlib interfaces\n");
  return success(not anyFatalErrors());
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
  auto cat = llvm::sys::findProgramByName("cat");
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
  auto formatCmd = llvm::join(args, " ");
  auto ret_code = system(formatCmd.c_str());
  if (ret_code != 0) {
    llvm::errs() << "ocamlformat failed with code " << ret_code << ": "
                 << ret_code << '\n';
    module->decl(os) << '\n';
    return;
  }

  // Then handle display based on color settings
  if (!llvm::sys::Process::StandardOutIsDisplayed() or !CL::Color) {
    std::stringstream ss;
    ss << *cat << ' ' << tmpFileName;
    auto rc = system(ss.str().c_str());
    if (rc != 0) {
      os << "Failed to read file: " << tmpFileName << '\n';
      module->decl(os) << '\n';
    }
    return;
  }

  std::vector<std::string> highlightArgs = {*bat, "-lml", "--plain", "--color", "always", tmpFileName};
  auto highlightCmd = llvm::join(highlightArgs, " ");
  ret_code = system(highlightCmd.c_str());
  if (ret_code != 0) {
    llvm::errs() << "bat highlighting failed with code " << ret_code << ": " << ret_code << '\n';
    module->decl(os) << '\n';
  }
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

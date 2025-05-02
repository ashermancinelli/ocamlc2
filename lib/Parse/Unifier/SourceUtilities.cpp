
#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <llvm/Support/FileSystem.h>

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

void Unifier::saveInterfaceDecl(std::string interface) {
  TRACE();
  if (!CL::DumpTypes or isLoadingStdlib) {
    return;
  }
  DBGS("Saving interface declaration: " << interface << '\n');
  nodesToDump.push_back(interface);
}

void Unifier::maybeDumpTypes(Node node, TypeExpr *type) {
  static std::set<uintptr_t> seen;
  if (!CL::DumpTypes or isLoadingStdlib or seen.count(node.getID())) {
    return;
  }
  seen.insert(node.getID());
  if (node.getType() == "let_binding") {
    auto name = node.getNamedChild(0);
    if (name.getType() != "unit") {
      saveInterfaceDecl(SSWRAP("val " << getTextSaved(name) << " : "
                                      << *getType(node.getID())));
    }
  } else if (node.getType() == "value_specification" ||
             node.getType() == "external") {
    auto name = node.getNamedChild(0);
    saveInterfaceDecl(SSWRAP("val " << getTextSaved(name) << " : "
                                    << *getType(node.getID())));
  }
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
  isLoadingStdlib = true;
  DBGS("Loading stdlib interfaces\n");
  for (auto filepath : getStdlibOCamlInterfaceFiles(exe)) {
    loadSourceFile(filepath);
  }
  DBGS("Done loading stdlib interfaces\n");
  isLoadingStdlib = false;
}

void Unifier::dumpTypes(llvm::raw_ostream &os) {
  if (!diagnostics.empty()) {
    return showErrors();
  }
  for (auto node : nodesToDump) {
    os << node << '\n';
  }
}

bool Unifier::anyFatalErrors() const { return !diagnostics.empty(); }
void Unifier::showErrors() {
  for (auto diag : diagnostics) {
    llvm::errs() << diag << "\n";
  }
}

llvm::raw_ostream &Unifier::showType(llvm::raw_ostream &os, llvm::StringRef name) {
  name = stringArena.save(name);
  if (auto *type = maybeGetVariableType(name)) {
    os << name << " : " << *type;
  } else {
    os << "Type unknown for symbol: '" << name << "'\n";
  }
  return os;
}

detail::Scope::Scope(Unifier *unifier)
    : unifier(unifier), envScope(unifier->env()),
      concreteTypes(unifier->concreteTypes) {
  DBGS("open scope\n");
}
detail::Scope::~Scope() {
  DBGS("close scope\n");
  unifier->concreteTypes = std::move(concreteTypes);
}

Unifier::TypeVarEnvScope::TypeVarEnvScope(Unifier::TypeVarEnv &env) : scope(env) {
  DBGS("open type var env scope\n");
}

Unifier::TypeVarEnvScope::~TypeVarEnvScope() {
  DBGS("close type var env scope\n");
}

} // namespace ocamlc2

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

#define DEBUG_TYPE "Environment.cpp"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"
#include "PathUtilities.h"

namespace ocamlc2 {

void Unifier::pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> path) {
  DBGS("pushing module search path: " << joinDot(path) << '\n');
  moduleSearchPath.push_back(llvm::SmallVector<llvm::StringRef>(path));
}

void Unifier::pushModule(llvm::StringRef module, const bool shouldDeclare) {
  module = stringArena.save(module);
  DBGS("pushing module: " << module << '\n');
  auto *moduleOperator = create<ModuleOperator>(module);
  auto *enclosingModule = moduleStack.empty() ? nullptr : moduleStack.back();
  moduleMap[module] = moduleOperator;
  moduleStack.push_back(moduleOperator);
  if (enclosingModule && shouldDeclare) {
    enclosingModule->exportVariable(module, moduleOperator);
  }
}

void Unifier::popModuleSearchPath() {
  DBGS("popping module search path: " << joinDot(moduleStack.back()->getName()) << '\n');
  // moduleStack.pop_back();
  // todo????
}

ModuleOperator *Unifier::popModule() {
  DBGS("popping module: " << moduleStack.back()->getName() << '\n');
  return moduleStack.pop_back_val();
}

nullptr_t Unifier::error(std::string message, ts::Node node, const char* filename, unsigned long lineno) {
  DBGS(message << '\n');
  if (diagnostics.size() >= (size_t)maxErrors) return nullptr;
  message += SSWRAP("\n" << "at " << filename << ":" << lineno);
  diagnostics.emplace_back(DiagKind::Error, message, sources.back().filepath, node.getPointRange());
  return nullptr;
}

nullptr_t Unifier::error(std::string message, const char* filename, unsigned long lineno) {
  DBGS(message << '\n');
  if (diagnostics.size() >= (size_t)maxErrors) return nullptr;
  message += SSWRAP("\n" << "at " << filename << ":" << lineno);
  diagnostics.emplace_back(DiagKind::Error, message, sources.back().filepath, std::nullopt);
  return nullptr;
}

LogicalResult Unifier::initializeEnvironment() {
  DBGS("Initializing environment\n");
  static bool initialized = false;
  if (initialized) {
    DBGS("Environment already initialized\n");
    return success();
  }
  initialized = true;
  pushModule("Caml_basics");
  // We need to insert these directly because other type initializations require
  // varargs, wildcard, etc to define themselves.
  for (std::string_view name : {"int", "float", "bool", "string", "unit", "_", "•"}) {
    auto str = stringArena.save(name);
    DBGS("Declaring type operator in Caml_basics: " << str << '\n');
    declareType(str, createTypeOperator(str));
  }
  declareType("list", createTypeOperator("list", createTypeVariable()));
  declareType("array", createTypeOperator("array", createTypeVariable()));

  // ref behaves both as a type and a regular function in the tree sitter ast
  auto *refType = createTypeOperator("ref", createTypeVariable());
  declareType("ref", refType);
  declareVariable("ref", getFunctionType({refType->back(), refType}));
  declareVariable(":=", getFunctionType({refType, refType->back(), getUnitType()}));

  openModules.push_back(moduleStack.back());
  popModule();
  return success();
}

} // namespace ocamlc2

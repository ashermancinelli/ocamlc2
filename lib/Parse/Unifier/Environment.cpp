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

#define DEBUG_TYPE "environment"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"
#include "PathUtilities.h"

namespace ocamlc2 {

void Unifier::pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> path) {
  DBGS("pushing module search path: " << joinDot(path) << '\n');
  moduleSearchPath.push_back(llvm::SmallVector<llvm::StringRef>(path));
}

void Unifier::pushModule(llvm::StringRef module) {
  module = stringArena.save(module);
  DBGS("pushing module: " << module << '\n');
  auto *moduleOperator = create<ModuleOperator>(module);
  auto *enclosingModule = moduleStack.empty() ? nullptr : moduleStack.back();
  moduleMap[module] = moduleOperator;
  moduleStack.push_back(moduleOperator);
  if (enclosingModule) {
    enclosingModule->exportVariable(module, moduleOperator);
  }
}

void Unifier::popModuleSearchPath() {
  DBGS("popping module search path: " << joinDot(moduleStack.back()->getName()) << '\n');
  // moduleStack.pop_back();
  // todo????
}

void Unifier::popModule() {
  DBGS("popping module: " << moduleStack.back()->getName() << '\n');
  moduleStack.pop_back();
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
  return success();
  pushModule("Stdlib");

  // We need to insert these directly because other type initializations require
  // varargs, wildcard, etc to define themselves.
  for (std::string_view name : {"int", "float", "bool", "string", "unit", "_", "•"}) {
    auto str = stringArena.save(name);
    DBGS("Declaring type operator: " << str << '\n');
    typeEnv().insert(str, createTypeOperator(str));
  }
  // auto *varargs = create<VarargsOperator>();
  // declareType({varargs->getName()}, varargs);

  auto *T_bool = getBoolType();
  auto *T_float = getFloatType();
  auto *T_int = getIntType();
  auto *T_unit = getUnitType();
  auto *T_string = getStringType();
  auto *T1 = createTypeVariable(), *T2 = createTypeVariable();

  {
    for (auto arithmetic : {"+", "-", "*", "/", "%"}) {
      declareVariable(arithmetic, getFunctionType({T_int, T_int, T_int}));
      declareVariable(std::string(arithmetic) + ".",
              getFunctionType({T_float, T_float, T_float}));
    }
    for (auto comparison : {"=", "!=", "<", "<=", ">", ">="}) {
      declareVariable(comparison, getFunctionType({T1, T1, T_bool}));
    }
  }
  {
    auto *concatLHS = getFunctionType({T1, T2});
    auto *concatType = getFunctionType({concatLHS, T1, T2});
    declareVariable("@@", concatType);
  }
  popModule();

  {
    // Builtin constructors
    detail::ModuleScope ms{*this, "Option"};
    auto *Optional = getOptionalType();
    declareVariable("None", Optional);
    declareVariable("Some", getFunctionType({Optional->back(), Optional}));
  }
  {
    detail::ModuleScope ms{*this, "Printf"};
    declareVariable("printf", getFunctionType({T_string, getVarargsType(), T_unit}));
  }

  return success();
}

} // namespace ocamlc2

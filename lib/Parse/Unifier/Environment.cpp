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

void Unifier::pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> modules) {
  auto path = hashPath(modules);
  DBGS("pushing module search path: " << path << '\n');
  moduleSearchPath.push_back(stringArena.save(path));
}

void Unifier::pushModule(llvm::StringRef module) {
  DBGS("pushing module: " << module << '\n');
  currentModule.push_back(stringArena.save(module.str()));
  pushModuleSearchPath(currentModule);
}

void Unifier::popModuleSearchPath() {
  DBGS("popping module search path: " << moduleSearchPath.back() << '\n');
  moduleSearchPath.pop_back();
}

void Unifier::popModule() {
  DBGS("popping module: " << currentModule.back() << '\n');
  currentModule.pop_back();
  popModuleSearchPath();
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
  pushModuleSearchPath("Stdlib");

  // We need to insert these directly because other type initializations require
  // varargs, wildcard, etc to define themselves.
  for (std::string_view name : {"int", "float", "bool", "string", "unit", "_", "•"}) {
    auto str = stringArena.save(name);
    DBGS("Declaring type operator: " << str << '\n');
    typeEnv.insert(str, createTypeOperator(str));
  }
  auto *varargs = create<VarargsOperator>();
  declareType(varargs->getName(), varargs);

  pushModule("Stdlib");
  auto *T_bool = getBoolType();
  auto *T_float = getFloatType();
  auto *T_int = getIntType();
  auto *T_unit = getUnitType();
  auto *T_string = getStringType();
  auto *T1 = createTypeVariable(), *T2 = createTypeVariable();

  popModule();

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
  {
    // Builtin constructors
    auto *Optional = getOptionalType();
    declareVariable("None", Optional);
    declareVariable("Some", getFunctionType({Optional->back(), Optional}));
  }
  declareVariablePath({"String", "concat"}, getFunctionType({T_string, getListTypeOf(T_string), T_string}));
  {
    detail::ModuleScope ms{*this, "Printf"};
    declareVariable("printf", getFunctionType({T_string, getVarargsType(), T_unit}));
  }
  return success();
}

} // namespace ocamlc2

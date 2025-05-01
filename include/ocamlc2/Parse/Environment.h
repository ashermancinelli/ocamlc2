#pragma once

#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Parse/ASTPasses.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/Utils.h"
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <set>
#include <filesystem>
#include <llvm/Support/Casting.h>
#include <ocamlc2/Parse/ScopedHashTable.h>
#include <cpp-tree-sitter.h>

namespace ocamlc2 {
namespace fs = std::filesystem;
struct TypeVariable;
struct TypeExpr;

using ts::Node;
using ts::NodeID;
using ts::Cursor;

using Env = llvm::ScopedHashTable<llvm::StringRef, TypeExpr *>;
using TypeVarEnv = llvm::ScopedHashTable<llvm::StringRef, TypeVariable *>;

struct TypeVarEnvScope {
  using ScopeTy = TypeVarEnv::ScopeTy;
  TypeVarEnvScope(TypeVarEnv &env);
  ~TypeVarEnvScope();

private:
  ScopeTy scope;
};
using EnvScope = Env::ScopeTy;

} // namespace ocamlc2

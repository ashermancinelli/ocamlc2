#include "ocamlc2/Parse/TypeSystem.h"

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

TypeExpr *ModuleOperator::lookupType(llvm::ArrayRef<llvm::StringRef> path) {
  DBGS("lookupType: " << llvm::join(path, ".") << '\n');
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = typeEnv.lookup(path.front())) {
    DBGS("found type: " << *type << '\n');
    if (auto *module = llvm::dyn_cast<ModuleOperator>(type)) {
      DBGS("found module: " << *module << '\n');
      return module->lookupType(path.drop_front());
    }
    return type;
  }
  DBGS("type not found\n");
  return nullptr;
}

TypeExpr *ModuleOperator::lookupVariable(llvm::StringRef name) {
  DBGS("lookupVariable: " << name << '\n');
  return variableEnv.lookup(name);
}

TypeExpr *ModuleOperator::lookupVariable(llvm::ArrayRef<llvm::StringRef> path) {
  DBGS("lookupVariable: " << llvm::join(path, ".") << '\n');
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = variableEnv.lookup(path.front())) {
    DBGS("found type: " << *type << '\n');
    if (auto *module = llvm::dyn_cast<ModuleOperator>(type)) {
      DBGS("found module: " << *module << '\n');
      return module->lookupVariable(path.drop_front());
    }
    return type;
  }
  DBGS("variable not found\n");
  return nullptr;
}

} // namespace ocamlc2

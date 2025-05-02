#include "ocamlc2/Parse/TypeSystem.h"

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

TypeExpr *ModuleOperator::lookupType(llvm::StringRef name) const {
  DBGS("lookupType: " << name << '\n');
  if (auto *type = typeEnv.lookup(name)) {
    return type;
  }
  for (auto *module : openModules) {
    DBGS("checking open module: " << module->getName() << '\n');
    if (auto *type = module->lookupType(name)) {
      DBGS("found type: " << *type << '\n');
      return type;
    }
  }
  return nullptr;
}

TypeExpr *ModuleOperator::lookupType(llvm::ArrayRef<llvm::StringRef> path) const {
  DBGS("lookupType: " << llvm::join(path, ".") << '\n');
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = lookupType(path.front())) {
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

TypeExpr *ModuleOperator::lookupVariable(llvm::StringRef name) const {
  DBGS("lookupVariable: " << name << '\n');
  if (auto *type = variableEnv.lookup(name)) {
    return type;
  }
  for (auto *module : openModules) {
    DBGS("checking open module: " << module->getName() << '\n');
    if (auto *type = module->lookupVariable(name)) {
      DBGS("found variable: " << *type << '\n');
      return type;
    }
  }
  DBGS("variable not found\n");
  return nullptr;
}

TypeExpr *ModuleOperator::lookupVariable(llvm::ArrayRef<llvm::StringRef> path) const {
  DBGS("lookupVariable: " << llvm::join(path, ".") << '\n');
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = lookupVariable(path.front())) {
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ModuleOperator &module) {
  os << "module " << module.getName() << " : sig\n";
  for (auto e : module.getExports()) {
    if (auto *type = module.lookupType(e)) {
      os << "type " << e << " = " << *type << '\n';
    } else if (auto *variable = module.lookupVariable(e)) {
      os << "val " << e << " : " << *variable << '\n';
    } else {
      assert(false && "unknown export");
    }
  }
  os << "end\n";
  return os;
}

} // namespace ocamlc2

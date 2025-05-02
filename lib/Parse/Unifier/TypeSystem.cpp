#include "ocamlc2/Parse/TypeSystem.h"

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

TypeExpr *SignatureOperator::lookupType(llvm::StringRef name) const {
  DBGS("lookupType: " << name << '\n');
  if (auto *type = typeEnv.lookup(name)) {
    return type;
  }
  return nullptr;
}

TypeExpr *SignatureOperator::lookupType(llvm::ArrayRef<llvm::StringRef> path) const {
  DBGS("lookupType: " << llvm::join(path, ".") << '\n');
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = lookupType(path.front())) {
    DBGS("found type: " << *type << '\n');
    if (auto *module = llvm::dyn_cast<SignatureOperator>(type)) {
      DBGS("found module: " << *module << '\n');
      if (path.size() == 1) {
        return module;
      }
      return module->lookupType(path.drop_front());
    }
    return type;
  }
  DBGS("type not found\n");
  return nullptr;
}

TypeExpr *SignatureOperator::lookupVariable(llvm::StringRef name) const {
  DBGS("lookupVariable: " << name << '\n');
  if (auto *type = variableEnv.lookup(name)) {
    DBGS("found variable: " << *type << '\n');
    return type;
  }
  DBGS("variable not found\n");
  return nullptr;
}

TypeExpr *SignatureOperator::lookupVariable(llvm::ArrayRef<llvm::StringRef> path) const {
  DBGS("lookupVariable: " << llvm::join(path, ".") << '\n');
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = lookupVariable(path.front())) {
    DBGS("found type: " << *type << '\n');
    if (auto *module = llvm::dyn_cast<SignatureOperator>(type)) {
      DBGS("found module: " << *module << '\n');
      if (path.size() == 1) {
        return module;
      }
      return module->lookupVariable(path.drop_front());
    }
    return type;
  }
  DBGS("variable not found\n");
  return nullptr;
}

TypeExpr *ModuleOperator::lookupType(llvm::StringRef name) const {
  TRACE();
  if (auto *type = SignatureOperator::lookupType(name)) {
    return type;
  }
  for (auto *module : openModules) {
    if (auto *type = module->lookupType(name)) {
      return type;
    }
  }
  return nullptr;
}

TypeExpr *ModuleOperator::lookupType(llvm::ArrayRef<llvm::StringRef> path) const {
  TRACE();
  if (auto *type = SignatureOperator::lookupType(path)) {
    return type;
  }
  for (auto *module : openModules) {
    if (auto *type = module->lookupType(path)) {
      return type;
    }
  }
  return nullptr;
}

TypeExpr *ModuleOperator::lookupVariable(llvm::StringRef name) const {
  TRACE();
  if (auto *type = SignatureOperator::lookupVariable(name)) {
    return type;
  }
  for (auto *module : openModules) {
    if (auto *type = module->lookupVariable(name)) {
      return type;
    }
  }
  return nullptr;
}

TypeExpr *ModuleOperator::lookupVariable(llvm::ArrayRef<llvm::StringRef> path) const {
  TRACE();
  if (auto *type = SignatureOperator::lookupVariable(path)) {
    return type;
  }
  for (auto *module : openModules) {
    if (auto *type = module->lookupVariable(path)) {
      return type;
    }
  }
  return nullptr;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const SignatureOperator &signature) {
  os << (signature.getKind() == TypeOperator::Kind::Module ? "module"
                                                        : "module type")
     << ' ' << signature.getName() << " = sig\n";
  
  for (auto e : signature.getExports()) {
    switch (e.kind) {
    case SignatureOperator::Export::Type: {
      if (auto *variantOperator = llvm::dyn_cast<VariantOperator>(e.type)) {
        os << variantOperator->decl() << '\n';
      } else if (auto *recordOperator = llvm::dyn_cast<RecordOperator>(e.type)) {
        os << recordOperator->decl() << '\n';
      } else {
        os << "type " << e.name << " = " << *e.type << '\n';
      }
      break;
    }
    case SignatureOperator::Export::Variable: {
      if (auto *module = llvm::dyn_cast<ModuleOperator>(e.type)) {
        os << *module;
      } else {
        os << "val " << e.name << " : " << *e.type << '\n';
      }
      break;
    }
    default:
      assert(false && "unknown export kind");
    }
  }
  os << "end\n";
  return os;
}

std::string VariantOperator::decl() const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "type ";
  for (auto typeArg : getArgs()) {
    ss << *typeArg << " ";
  }
  ss << getName() << " = ";
  auto first = constructors.front();
  showCtor(ss, first);
  for (auto ctor : llvm::drop_begin(constructors)) {
    ss << " | ";
    showCtor(ss, ctor);
  }
  return s;
}

} // namespace ocamlc2

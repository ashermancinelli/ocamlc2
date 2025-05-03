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
     << ' ' << signature.getName() << " : sig\n";
  
  // eg type constructors will show up as functions but we don't really want to print those
  llvm::SmallVector<llvm::StringRef> namesToSkip;
  for (auto e : signature.getExports()) {
    if (std::find(namesToSkip.begin(), namesToSkip.end(), e.name) != namesToSkip.end()) {
      continue;
    }
    // only dump once - for type aliases they may show up again
    // namesToSkip.push_back(e.name);
    switch (e.kind) {
    case SignatureOperator::Export::Type: {
      if (auto *variantOperator = llvm::dyn_cast<VariantOperator>(e.type)) {
        os << variantOperator->decl() << '\n';
        for (auto ctor : variantOperator->getConstructorNames()) {
          namesToSkip.push_back(ctor);
        }
      } else if (auto *recordOperator = llvm::dyn_cast<RecordOperator>(e.type)) {
        os << recordOperator->decl(true) << '\n';
      } else if (auto *to = llvm::dyn_cast<TypeOperator>(e.type)) {
        os << "type ";
        if (auto *to = llvm::dyn_cast<TypeOperator>(e.type)) {
          for (auto *arg : to->getArgs()) {
            if (TypeVariable *tv = llvm::dyn_cast<TypeVariable>(arg)) {
              if (!tv->instantiated()) {
                os << *tv << " ";
              }
            } else {
              os << *arg << " ";
            }
          }
        }
        os << e.name;
        if (to->getArgs().empty() and e.name == to->getName()) {
          // just a decl, maybe don't show anything else? eg `type t`
        } else {
          os << " = " << *e.type;
        }
        os << '\n';
      } else {
        assert(false && "unknown type operator");
      }
      break;
    }
    case SignatureOperator::Export::Variable: {
      if (auto *module = llvm::dyn_cast<ModuleOperator>(e.type)) {
        os << *module << '\n';
      } else if (auto *fo = llvm::dyn_cast<FunctionOperator>(e.type)) {
        os << "val " << e.name << " : " << *fo << '\n';
      } else if (auto *to = llvm::dyn_cast<TypeOperator>(e.type)) {
        os << "val " << e.name << " : ";
        if (!to->getArgs().empty()) {
          auto *arg = to->getArgs().front();
          if (to->getArgs().size() > 1) {
            os << "(" << *arg;
            for (auto *arg : llvm::drop_begin(to->getArgs())) {
              os << ", " << *arg;
            }
            os << ") ";
          } else {
            os << *arg << " ";
          }
        }
        os << to->getName() << '\n';
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

std::string RecordOperator::decl(const bool named) const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  auto zipped = llvm::zip(fieldNames, fieldTypes);
  if (named) {
    ss << "type ";
    for (auto *arg : getArgs()) {
      ss << *arg << " ";
    }
    ss << getName() << " = ";
  }
  ss << "{ ";
  for (auto [name, type] : zipped) {
    ss << name << " : " << *type << "; ";
  }
  ss << '}';
  return ss.str();
}

} // namespace ocamlc2

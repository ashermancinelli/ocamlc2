#include "ocamlc2/Parse/TypeSystem.h"
#include <llvm/ADT/STLExtras.h>

#define DEBUG_TYPE "TypeSystem.cpp"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

TypeExpr *SignatureOperator::lookupType(llvm::StringRef name) const {
  DBGS("lookupType: " << name << '\n');
  if (auto *type = typeEnv.lookup(name)) {
    return type;
  }
  DBGS("type not found\n");
  return nullptr;
}

TypeExpr *SignatureOperator::lookupType(llvm::ArrayRef<llvm::StringRef> path) const {
  DBGS("lookupType: " << llvm::join(path, ".") << '\n');
  DBGS("Locals: " << locals.size() << " exports: " << exports.size() << '\n');
  auto front = path.front();
  assert(not path.empty() && "module path cannot be empty");
  if (auto *type = lookupType(front)) {
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
  DBGS("Look up path part " << front
                            << " as a variable in case it's a module\n");
  if (auto *type = lookupVariable(front)) {
    DBGS("found type when lookup up the first part of type path: " << *type << '\n');
    if (auto *module = llvm::dyn_cast<SignatureOperator>(type)) {
      assert(path.size() != 1 && "module path cannot be a single module name");
      DBGS("found module: " << *module << '\n');
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

TypeExpr *SignatureOperator::exportType(llvm::StringRef name, TypeExpr *type) {
  DBGS("exportType: " << name << " : " << *type << '\n');
  typeEnv.insert(name, type);
  if (typeEnv.getCurScope() == &rootTypeScope) {
    auto found = llvm::find_if(exports, [&](const auto &e) {
      return e.name == name && e.kind == Export::Kind::Type;
    });
    auto exported = Export(Export::Kind::Type, name, type);
    if (found == exports.end()) {
      exports.push_back(exported);
    } else {
      *found = exported;
    }
  }
  return type;
}

TypeExpr *SignatureOperator::exportVariable(llvm::StringRef name,
                                            TypeExpr *type) {
  DBGS("exportVariable: " << name << " : " << *type << '\n');
  variableEnv.insert(name, type);
  if (variableEnv.getCurScope() == &rootVariableScope) {
    auto found = llvm::find_if(exports, [&](const auto &e) {
      return e.name == name && e.kind == Export::Kind::Variable;
    });
    auto exported = Export(Export::Kind::Variable, name, type);
    if (found == exports.end()) {
      exports.push_back(exported);
    } else {
      *found = exported;
    }
  }
  return type;
}

TypeExpr *SignatureOperator::localType(llvm::StringRef name, TypeExpr *type) {
  DBGS("localType: " << name << " : " << *type << '\n');
  typeEnv.insert(name, type);
  locals.push_back(Export(Export::Kind::Type, name, type));
  DBGS("Locals: " << locals.size() << " exports: " << exports.size() << '\n');
  return type;
}

TypeExpr *SignatureOperator::localVariable(llvm::StringRef name,
                                           TypeExpr *type) {
  DBGS("localVariable: " << name << " : " << *type << '\n');
  variableEnv.insert(name, type);
  locals.push_back(Export(Export::Kind::Variable, name, type));
  DBGS("Locals: " << locals.size() << " exports: " << exports.size() << '\n');
  return type;
}

llvm::raw_ostream &SignatureOperator::showSignature(llvm::raw_ostream &os) const {
  // eg type constructors will show up as functions but we don't really want to print those
  llvm::SmallVector<llvm::StringRef> namesToSkip;
  for (auto e : getExports()) {
    if (std::find_if(namesToSkip.begin(), namesToSkip.end(), [&](const auto &other) {
          return other == e.name;
        }) != namesToSkip.end()) {
      continue;
    }
    switch (e.kind) {
    case SignatureOperator::Export::Type: {
      if (auto *variantOperator = llvm::dyn_cast<VariantOperator>(e.type)) {
        os << variantOperator->decl() << SignatureOperator::newline;
        for (auto ctor : variantOperator->getConstructorNames()) {
          namesToSkip.push_back(ctor);
        }
      } else if (auto *recordOperator = llvm::dyn_cast<RecordOperator>(e.type)) {
        os << recordOperator->decl(true) << SignatureOperator::newline;
      } else if (auto *to = llvm::dyn_cast<TypeOperator>(e.type)) {
        os << "type ";
        if (auto *to = llvm::dyn_cast<TypeOperator>(e.type)) {
          for (auto *arg : to->getArgs()) {
            os << *arg << " ";
          }
        }
        if (not e.name.empty()) {
          os << e.name;
        }
        if (to->getArgs().empty() and e.name == to->getName()) {
          // just a decl, maybe don't show anything else? eg `type t`
        } else {
          os << " = " << *e.type;
        }
        os << SignatureOperator::newline;
      } else if (llvm::isa<TypeVariable>(e.type)) {
        os << "type " << e.name << " = " << *e.type << SignatureOperator::newline;
      } else {
        os << "unknown type operator: " << e.name << *e.type << SignatureOperator::newline;
        assert(false && "unknown type operator");
      }
      break;
    }
    case SignatureOperator::Export::Variable: {
      if (auto *module = llvm::dyn_cast<ModuleOperator>(e.type)) {
        if (auto *sig = module->getInterfaceSignature()) {
          os << "module " << e.name << " : ";
          if (sig->getName() != SignatureOperator::getAnonymousSignatureName()) {
            os << sig->getName();
          } else {
            os << "sig" << SignatureOperator::newline;
            sig->showSignature(os);
            os << "end";
          }
        } else {
          os << *module;
        }
        os << SignatureOperator::newline;
      } else if (auto *functor = llvm::dyn_cast<FunctorOperator>(e.type)) {
        os << *functor << SignatureOperator::newline;
      } else if (auto *fo = llvm::dyn_cast<FunctionOperator>(e.type)) {
        os << "val " << e.name << " : " << *fo << SignatureOperator::newline;
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
        os << to->getName() << SignatureOperator::newline;
      } else {
        os << "val " << e.name << " : " << *e.type << SignatureOperator::newline;
      }
      break;
    }
    default:
      assert(false && "unknown export kind");
    }
  }
  return os;
}

llvm::raw_ostream &SignatureOperator::showDeclaration(llvm::raw_ostream &os) const {
  os << (isModuleType() ? "module type" : "module")
     << ' ' << getName() << " : sig" << SignatureOperator::newline;
  showSignature(os);
  os << "end";
  return os;
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

char SignatureOperator::newline = ' ';

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SignatureOperator &signature) {
  return signature.showDeclaration(os);
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FunctorOperator &functor) {
  auto *functorResult = functor.back();
  // auto *sig = llvm::dyn_cast<SignatureOperator>(functor.back());
  auto showSig = [&] {
    // if (auto *interfaceSig = functor.getInterfaceSignature()) {
    //   os << " -> " << interfaceSig->getName();
    // } else if (sig) {
    //   sig->showSignature(os);
    // } else {
      os << *functorResult << SignatureOperator::newlineCharacter();
    // }
  };
  if (functor.getModuleParameters().empty()) {
    os << "functor";
    for (auto *arg : llvm::drop_end(functor.getArgs())) {
      os << " (" << *arg << ")";
    }
    os << " -> ";
    showSig();
  } else {
    os << "module " << functor.getName();
    for (auto [param, type] : llvm::zip(functor.getModuleParameters(), functor.getArgs())) {
      auto [paramName, _] = param;
      os << " (" << paramName << " : " << *type << ")";
    }
    os << " : sig" << SignatureOperator::newlineCharacter();
    showSig();
    os << "end";
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeExpr &type) {
  if (auto *fo = llvm::dyn_cast<FunctionOperator>(&type)) {
    os << *fo;
  } else if (auto *vo = llvm::dyn_cast<VariantOperator>(&type)) {
    os << *vo;
  } else if (auto *to = llvm::dyn_cast<TupleOperator>(&type)) {
    os << *to;
  } else if (auto *ro = llvm::dyn_cast<RecordOperator>(&type)) {
    os << *ro;
  } else if (auto *mo = llvm::dyn_cast<ModuleOperator>(&type)) {
    os << *mo;
  } else if (auto *fo = llvm::dyn_cast<FunctorOperator>(&type)) {
    os << *fo;
  } else if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    os << *to;
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(&type)) {
    os << *tv;
  }
  return os;
}

llvm::raw_ostream &VariantOperator::showCtor(llvm::raw_ostream &os, const ConstructorType &ctor) const {
  if (std::holds_alternative<llvm::StringRef>(ctor)) {
    os << std::get<llvm::StringRef>(ctor);
  } else {
    auto [name, fo] =
        std::get<std::pair<llvm::StringRef, FunctionOperator *>>(ctor);
    os << name << " of ";
    if (fo->getArgs().size() == 2) {
      os << *fo->getArgs().front();
    } else {
      os << "(" << *fo->getArgs().front();
      for (auto *arg : fo->getArgs().drop_front().drop_back()) {
        os << " * " << *arg;
      }
      os << ")";
    }
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FunctionOperator &func) {
  const auto argList = func.getArgs();
  const auto *returnType = argList.back();
  if (!returnType)
    return os << "<error>";
  const auto args = argList.drop_back();
  assert(!args.empty() && "function type must have at least one argument");
  const auto &descs = func.parameterDescriptors;
  assert(args.size() == descs.size() &&
         "argument list and parameter descriptors must be the same size");
  auto argIter = llvm::zip(descs, args);
  os << '(';
  auto showArg = [&](auto desc, auto *arg) -> llvm::raw_ostream & {
    if (desc.isOptional()) {
      os << "?";
    }
    if (desc.isNamed()) {
      os << desc.label.value() << ":";
    }
    return os << *arg;
  };
  for (auto [desc, arg] : argIter) {
    if (!arg)
      return os << "<error>";
    showArg(desc, arg) << " -> ";
  }
  if (!returnType)
    return os << "<error>";
  os << *returnType;
  return os << ')';
}

} // namespace ocamlc2

#include "ocamlc2/Parse/TypeSystem.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include "ocamlc2/Parse/TSUnifier.h"
#define DEBUG_TYPE "TypeSystem.cpp"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

static llvm::raw_ostream &showTypeVariables(llvm::raw_ostream &os, const TypeOperator &type) {
  auto vars = SmallVector<TypeExpr *>(type.getArgs());
  switch (vars.size()) {
  case 0:
    break;
  case 1:
    os << *vars.front();
    break;
  default: {
    os << '(' << *vars.front();
    for (auto *arg : llvm::drop_begin(vars)) {
      os << ", " << *arg;
    }
    os << ')';
  }
  }
  return os << ' ';
}

[[maybe_unused]] static llvm::raw_ostream &showUninstantiatedTypeVariables(llvm::raw_ostream &os, const TypeOperator &type) {
  SmallVector<const TypeExpr *> vars;
  for (auto *arg : type.getArgs()) {
    if (isa::uninstantiatedTypeVariable(arg)) {
      vars.push_back(arg);
    }
  }
  switch (vars.size()) {
  case 0:
    break;
  case 1:
    os << *vars.front();
    break;
  default: {
    os << '(' << *vars.front();
    for (auto *arg : llvm::drop_begin(vars)) {
      os << ", " << *arg;
    }
    os << ')';
  }
  }
  return os << ' ';
}

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

llvm::raw_ostream &TypeAlias::decl(llvm::raw_ostream &os) const {
  os << "type ";
  if (auto *to = llvm::dyn_cast<TypeOperator>(getType())) {
    for (auto *arg : to->getArgs()) {
      if (isa::uninstantiatedTypeVariable(arg)) {
        os << *arg << " ";
      }
    }
  }
  os << getName();
  auto *type = Unifier::pruneTypeVariables(getType());
  if (isa::uninstantiatedTypeVariable(type)) {
    return os;
  } else if (auto *alias = llvm::dyn_cast<TypeAlias>(type)) {
    os << " = " << alias->getName();
  } else {
    os << " = " << *type;
  }
  // os << " (alias " << *this << ")";
  return os;
}

static llvm::StringRef escape(llvm::StringRef str) {
  static constexpr llvm::StringRef namesToWrapInParens[]{
      "^",  "let*", "let+", "+",  "-",  "*",  "/",  "%",  "<",
      "*.", "/.",   "<.",   "+.", "-.", "<=", ">",  ">=", "=",
      "<>", "&&",   "||",   "@@", "^",  "=",  "<>", "<",
  };
  static std::vector<std::string> saved;
  if (llvm::is_contained(namesToWrapInParens, str)) {
    saved.push_back("( " + str.str() + " )");
    return saved.back();
  }
  return str;
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
    auto *exportedType = Unifier::pruneTypeVariables(e.type);
    switch (e.kind) {
    case SignatureOperator::Export::Type: {
      if (auto *variantOperator = llvm::dyn_cast<VariantOperator>(exportedType)) {
        variantOperator->decl(os) << SignatureOperator::newline;
        for (auto ctor : variantOperator->getConstructorNames()) {
          namesToSkip.push_back(ctor);
        }
      } else if (auto *recordOperator = llvm::dyn_cast<RecordOperator>(exportedType)) {
        recordOperator->decl(os, true) << SignatureOperator::newline;
      } else if (auto *to = llvm::dyn_cast<TypeOperator>(exportedType)) {
        os << "type " << e.name << " = " << *to << SignatureOperator::newline;
        // ::ocamlc2::decl(os, *to) << SignatureOperator::newline;
      } else if (auto *alias = llvm::dyn_cast<TypeAlias>(exportedType)) {
        os << "type ";
        auto *type = Unifier::pruneTypeVariables(alias->getType());
        if (auto *to = llvm::dyn_cast<TypeOperator>(type)) {
          showUninstantiatedTypeVariables(os, *to);
          os << ' ' << e.name << " = " << *to;
        } else {
          os << ' ' << e.name;
          if (!isa::uninstantiatedTypeVariable(type)) {
            os << " = " << *type;
          }
        }
        os << SignatureOperator::newline;
        // alias->decl(os) << SignatureOperator::newlineCharacter();
      } else if (llvm::isa<TypeVariable>(exportedType)) {
        os << "type " << e.name << " = " << *exportedType << SignatureOperator::newline;
      } else {
        os << "unknown type operator: " << e.name << *exportedType << SignatureOperator::newline;
        assert(false && "unknown type operator");
      }
      break;
    }
    case SignatureOperator::Export::Variable: {
      auto name = escape(e.name);
      if (auto *module = llvm::dyn_cast<ModuleOperator>(exportedType)) {
        module->decl(os) << SignatureOperator::newline;
      } else if (auto *functor = llvm::dyn_cast<FunctorOperator>(exportedType)) {
        functor->decl(os) << SignatureOperator::newline;
      } else if (auto *fo = llvm::dyn_cast<FunctionOperator>(exportedType)) {
        os << "val " << name << " : " << *fo << SignatureOperator::newline;
      } else if (auto *to = llvm::dyn_cast<TypeOperator>(exportedType)) {
        os << "val " << name << " : ";
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
        os << "val " << name << " : " << *e.type << SignatureOperator::newline;
      }
      break;
    }
    default:
      assert(false && "unknown export kind");
    }
  }
  return os;
}

llvm::raw_ostream &SignatureOperator::decl(llvm::raw_ostream &os) const {
  if (!isAnonymous()) {
    os << (isModuleType() ? "module type" : "module") << ' ' << getName();
    os << (isModuleType() ? " = " : " : ");
  }
  os << "sig" << SignatureOperator::newline;
  showSignature(os);
  os << "end";
  return os;
}

llvm::raw_ostream &ModuleOperator::decl(llvm::raw_ostream &os) const {
  return SignatureOperator::decl(os);
}

llvm::raw_ostream &FunctorOperator::decl(llvm::raw_ostream &os) const {
  auto name = getName();
  auto args = getArgs().drop_back();
  auto params = getModuleParameters();
  auto *resultSignature = back();
  os << "module " << name << " :";
  for (auto [param, arg] : llvm::zip_longest(params, args)) {
    auto *argExpr = *arg;
    if (param) {
      os << " (" << param->first << " : " << argExpr->getName() << ")";
    } else {
      os << " (" << *argExpr << ")";
    }
  }
  os << " -> " << *resultSignature;
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeAlias &alias) {
  // return os << "alias " << alias.getName() << " = " << *alias.getType();
  // if (auto *to = llvm::dyn_cast<TypeOperator>(alias.getType())) {
  //   showUninstantiatedTypeVariables(os, *to);
  // }
  return os << alias.getName();
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
  return signature.decl(os);
}

llvm::raw_ostream &VariantOperator::decl(llvm::raw_ostream &os) const {
  os << "type ";
  for (auto typeArg : getArgs()) {
    os << *typeArg << " ";
  }
  os << getName() << " = ";
  auto first = constructors.front();
  showCtor(os, first);
  for (auto ctor : llvm::drop_begin(constructors)) {
    os << " | ";
    showCtor(os, ctor);
  }
  return os;
}

llvm::raw_ostream &RecordOperator::decl(llvm::raw_ostream &os, const bool named) const {
  auto zipped = llvm::zip(fieldNames, fieldTypes);
  if (named) {
    os << "type ";
    showTypeVariables(os, *this);
    os << getName() << " = ";
  }
  os << "{ ";
  for (auto [name, type] : zipped) {
    os << name << " : " << *type << "; ";
  }
  os << '}';
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FunctorOperator &functor) {
  auto *functorResult = functor.back();
  auto functorParams = functor.getArgs().drop_back();
  os << "functor";
  if (functor.getModuleParameters().empty()) {
    for (auto *arg : functorParams) {
      os << " (" << *arg << ")";
    }
  } else {
    for (auto [param, type] : llvm::zip(functor.getModuleParameters(), functorParams)) {
      os << " (" << param.first << " : " << *type << ")";
    }
  }
  if (functor.getInterfaceSignature()) {
    os << " : " << functor.getInterfaceSignature()->getName();
  }
  return os << " -> " << *functorResult << SignatureOperator::newlineCharacter();
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
  } else if (auto *alias = llvm::dyn_cast<TypeAlias>(&type)) {
    os << *alias;
  } else {
    assert(false && "unknown type");
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


static llvm::raw_ostream &decl(llvm::raw_ostream &os, const TypeOperator &type) {
  os << "type ";
  const auto arity = type.size();
  if (arity > 1) {
    os << "(";
    for (auto [i, rawArg] : llvm::enumerate(type.getArgs())) {
      auto *arg = Unifier::pruneTypeVariables(rawArg);
      if (isa::uninstantiatedTypeVariable(arg)) {
        os << *arg << (i == arity - 1 ? "" : ", ");
      }
    }
    os << ") ";
  } else if (arity == 1) {
    os << type.getArgs().front() << ' ';
  }
  os << type.getName();
  os << " = " << type;
  return os;
}

llvm::raw_ostream &decl(llvm::raw_ostream &os, const TypeExpr &type) {
  if (auto *ro = llvm::dyn_cast<RecordOperator>(&type)) {
    return ro->decl(os);
  } else if (auto *so = llvm::dyn_cast<SignatureOperator>(&type)) {
    return so->decl(os);
  } else if (auto *vo = llvm::dyn_cast<VariantOperator>(&type)) {
    return vo->decl(os);
  } else if (auto *mo = llvm::dyn_cast<ModuleOperator>(&type)) {
    return mo->decl(os);
  } else if (auto *fo = llvm::dyn_cast<FunctorOperator>(&type)) {
    return fo->decl(os);
  } else if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    return decl(os, *to);
  } else if (auto *ta = llvm::dyn_cast<TypeAlias>(&type)) {
    return ta->decl(os);
  } else {
    return os << type;
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeOperator &op) {
  auto args = op.getArgs();
  auto name = op.getName().str();
  if (args.empty()) {
    return os << name;
  }
  if (args.size() == 1) {
    return os << *args.front() << ' ' << name;
  }
  os << '(' << *args.front();
  for (auto *arg : args.drop_front()) {
    os << ", " << *arg;
  }
  return os << ") " << name;
}

} // namespace ocamlc2

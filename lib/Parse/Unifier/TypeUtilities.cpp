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
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <llvm/Support/FileSystem.h>

#define DEBUG_TYPE "TypeUtilities"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"
#include "PathUtilities.h"

namespace ocamlc2 {

namespace {
static constexpr std::string_view pathTypes[] = {
    "value_path",
    "module_path",
    "constructor_path",
    "type_constructor_path",
};
}

bool Unifier::isVarargs(TypeExpr* type) {
  return llvm::isa<VarargsOperator>(type);
}

bool Unifier::isWildcard(TypeExpr* type) {
  return llvm::isa<WildcardOperator>(type);
}

TypeExpr *Unifier::getBoolType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getBoolOperatorName());
  return type;
}

TypeExpr *Unifier::getFloatType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getFloatOperatorName());
  return type;
}

TypeExpr *Unifier::getIntType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getIntOperatorName());
  return type;
}

TypeExpr *Unifier::getUnitType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getUnitOperatorName());
  return type;
}

TypeExpr *Unifier::getStringType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getStringOperatorName());
  return type;
}

TypeExpr *Unifier::getWildcardType() { 
  static TypeExpr *type = create<WildcardOperator>();
  return type;
}

TypeExpr *Unifier::getVarargsType() { 
  static TypeExpr *type = create<VarargsOperator>();
  return type;
}

llvm::SmallVector<llvm::StringRef> Unifier::getPathParts(Node node) {
  llvm::SmallVector<llvm::StringRef> pathParts;
  static constexpr std::string_view nameTypes[] = {
    "value_name",
    "module_name",
    "constructor_name",
    "type_constructor",
    "type_variable",
  };
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    auto childType = child.getType();
    if (llvm::is_contained(pathTypes, childType)) {
      auto parts = getPathParts(child);
      pathParts.append(parts.begin(), parts.end());
    } else if (llvm::is_contained(nameTypes, childType)) {
      llvm::StringRef part = stringArena.save(getText(child));
      pathParts.push_back(part);
    } else if (childType == "parenthesized_operator") {
      pathParts.push_back(stringArena.save(getText(child.getNamedChild(0))));
    } else {
      assert(false && "Unknown path part type");
    }
  }
  DBGS("Path parts: " << llvm::join(pathParts, "<join>") << '\n');
  return pathParts;
}

TypeExpr *Unifier::declareType(llvm::StringRef name, TypeExpr* type) {
  TRACE();
  ORNULL(type);
  typeEnv().insert(name, type);
  return type;
}

TypeExpr *Unifier::declareType(Node node, TypeExpr* type) {
  TRACE();
  ORNULL(type);
  return declareType(getTextSaved(node), type);
}

TypeExpr *Unifier::maybeGetDeclaredType(ArrayRef<llvm::StringRef> path) {
  DBGS("maybeGetDeclaredType: " << llvm::join(path, ".") << '\n');
  const auto sz = path.size();
  // first: look in current module's environment
  if (auto *type = typeEnv().lookup(path.front())) {
    DBGS("found in current module's environment: " << *type << '\n');
    if (sz == 1) {
      DBGS("returning type from current module's environment: " << *type << '\n');
      return clone(type);
    } else if (auto *module = llvm::dyn_cast<ModuleOperator>(type)) {
      DBGS("looking up type in current module: " << *module << '\n');
      return module->lookupType(path.drop_front());
    }
    assert(false && "hmm");
  }
  // TODO: search reverse module stack for vars
  for (auto &module : llvm::reverse(moduleStack)) {
    DBGS("checking module: " << module->getName() << '\n');
    if (auto *type = module->lookupType(path.front())) {
      DBGS("found in module: " << *type << '\n');
      if (sz == 1) {
        DBGS("returning type from module: " << *type << '\n');
        return clone(type);
      }
      return maybeGetDeclaredType(path.drop_front());
    }
  }
  // then fall back on search path (e.g. default path Stdlib or imported
  // List or other passed on command line)
  for (ArrayRef<llvm::StringRef> searchPath : llvm::reverse(moduleSearchPath)) {
    DBGS("checking search path: " << llvm::join(searchPath, ".") << '\n');
    if (auto *module = moduleMap.lookup(searchPath.front())) {
      DBGS("checking module: " << module->getName() << '\n');
      SmallVector<llvm::StringRef> remainingPath = SmallVector<StringRef>{searchPath.drop_front()};
      remainingPath.append(path.begin(), path.end());
      if (auto *type = module->lookupType(remainingPath)) {
        DBGS("found in search path: " << *type << '\n');
        return clone(type);
      }
    }
  }
  return nullptr;
}

TypeExpr *Unifier::getDeclaredType(ArrayRef<llvm::StringRef> path) {
  TRACE();
  if (auto *type = maybeGetDeclaredType(path)) {
    return type;
  }
  RNULL("Type not declared: " + llvm::join(path, "."));
}

TypeExpr *Unifier::getDeclaredType(Node node) {
  TRACE();
  auto path = getPathParts(node);
  return getDeclaredType(path);
}

TypeExpr* Unifier::declareVariable(llvm::StringRef name, TypeExpr* type) {
  TRACE();
  ORNULL(type);
  auto str = getHashedPathSaved({name});
  DBGS("Declaring: " << str << " as " << *type << '\n');
  if (name == getWildcardType()->getName() or name == getUnitType()->getName()) {
    DBGS("probably declaring a wildcard variable in a constructor pattern or "
         "assigning to unit, skipping\n");
    return type;
  }
  if (env().count(str)) {
    DBGS("WARNING: Type of " << name << " redeclared\n");
  }
  env().insert(str, type);
  return type;
}

TypeExpr* Unifier::declareVariablePath(llvm::ArrayRef<llvm::StringRef> path, TypeExpr* type) {
  TRACE();
  auto hashedPath = hashPath(path);
  DBGS("Declaring path: " << getPath(path) << '(' << hashedPath << ')' << " as " << *type << '\n');
  return declareVariable(hashedPath, type);
}

TypeExpr* Unifier::setType(Node node, TypeExpr *type) {
  nodeToType[node.getID()] = type;
  return type;
}

TypeExpr* Unifier::getType(ts::NodeID id) {
  return nodeToType.lookup(id);
}

TypeExpr* Unifier::getVariableType(const char *name) {
  return getVariableType(stringArena.save(name));
}

TypeExpr* Unifier::getVariableType(Node node) {
  DBGS("Getting type for: " << node.getType() << '\n');
  if (llvm::is_contained(pathTypes, node.getType())) {
    return getVariableType(getPathParts(node));
  }
  return getVariableType(getTextSaved(node));
}

TypeExpr* Unifier::getVariableType(const std::string_view name) {
  return getVariableType(stringArena.save(name));
}

TypeExpr* Unifier::getVariableType(llvm::SmallVector<llvm::StringRef> path) {
  auto hashedPath = hashPath(path);
  return getVariableType(stringArena.save(hashedPath));
}

TypeExpr* Unifier::getVariableType(const llvm::StringRef name) {
  TRACE();
  if (auto *type = maybeGetVariableType(name)) {
    return type;
  }
  RNULL("Type not declared: " + name.str());
}

TypeExpr* Unifier::maybeGetVariableType(const llvm::StringRef name) {
  DBGS("Getting type: " << name << '\n');
  for (auto &module : llvm::reverse(moduleStack)) {
    DBGS("Checking module: " << module->getName() << '\n');
    if (auto *type = module->lookupVariable(name)) {
      DBGS("Found type: " << *type << '\n');
      return clone(type);
    }
  }
  for (ArrayRef<llvm::StringRef> path : llvm::reverse(moduleSearchPath)) {
    DBGS("Checking module search path: " << llvm::join(path, ".") << '\n');
    if (auto *module = moduleMap.lookup(path.front())) {
      DBGS("Checking module: " << module->getName() << path.front() << '\n');
      if (auto *type = module->lookupVariable(path.drop_front())) {
        DBGS("Found type: " << *type << '\n');
        return clone(type);
      }
    }
  }
  DBGS("Type not declared: " << name << '\n');
  return nullptr;
}

TypeExpr *Unifier::maybeGetVariableTypeWithName(const llvm::StringRef name) {
  DBGS(name << '\n');
  if (auto *type = env().lookup(name)) {
    DBGS("Found type " << *type << '\n');
    return clone(type);
  }
  DBGS("not found\n");
  return nullptr;
}

TypeVariable* Unifier::declareTypeVariable(llvm::StringRef name) {
  auto str = stringArena.save(name);
  auto *type = createTypeVariable();
  typeVarEnv.insert(str, type);
  return type;
}

TypeVariable* Unifier::getTypeVariable(Node node) {
  TRACE();
  assert(node.getType() == "type_variable");
  auto text = getTextSaved(node);
  return getTypeVariable(text);
}

TypeVariable* Unifier::getTypeVariable(llvm::StringRef name) {
  DBGS("Getting type variable: " << name << '\n');
  auto str = stringArena.save(name);
  if (auto *type = typeVarEnv.lookup(str)) {
    DBGS("Found type variable: " << *type << '\n');
    return type;
  }
  DBGS("Type variable not found, declaring\n");
  return declareTypeVariable(name);
}

llvm::StringRef Unifier::getHashedPathSaved(llvm::ArrayRef<llvm::StringRef> path) {
  return stringArena.save(getHashedPath(path));
}

std::string Unifier::getHashedPath(llvm::ArrayRef<llvm::StringRef> path) {
  TRACE();
  return hashPath(path);
}

} // namespace ocamlc2

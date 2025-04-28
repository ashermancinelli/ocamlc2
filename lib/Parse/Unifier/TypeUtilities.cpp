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

#define DEBUG_TYPE "typeutilities"
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
  if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    return op->getName() == getWildcardType()->getName();
  }
  return false;
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
  static TypeExpr *type = getDeclaredType(TypeOperator::getWildcardOperatorName());
  return type;
}

TypeExpr *Unifier::getVarargsType() { 
  static TypeExpr *type = create<VarargsOperator>();
  return type;
}

std::vector<std::string> Unifier::getPathParts(Node node) {
  std::vector<std::string> pathParts;
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
      pathParts.insert(pathParts.end(), parts.begin(), parts.end());
    } else if (llvm::is_contained(nameTypes, childType)) {
      std::string part{getText(child)};
      pathParts.push_back(part);
    } else if (childType == "parenthesized_operator") {
      pathParts.push_back(std::string{getText(child.getNamedChild(0))});
    } else {
      assert(false && "Unknown path part type");
    }
  }
  DBGS("Path parts: " << llvm::join(pathParts, "<join>") << '\n');
  return pathParts;
}

TypeExpr *Unifier::declareType(Node node, TypeExpr* type) {
  TRACE();
  ORNULL(type);
  return declareType(getTextSaved(node), type);
}

TypeExpr *Unifier::declareType(llvm::StringRef name, TypeExpr* type) {
  TRACE();
  ORNULL(type);
  auto str = getHashedPathSaved({name});
  DBGS("Declaring type: " << str << " as " << *type << '\n');
  typeEnv.insert(str, type);
  return type;
}

TypeExpr *Unifier::maybeGetDeclaredType(llvm::StringRef name) {
  TRACE();
  if (auto *type = maybeGetDeclaredTypeWithName(name)) {
    return type;
  }
  for (auto &path : llvm::reverse(moduleSearchPath)) {
    auto possiblePath = hashPath(std::vector<std::string>{path.str(), name.str()});
    auto str = stringArena.save(possiblePath);
    DBGS("Checking module search path: " << path << " with name: " << str << '\n');
    if (auto *type = maybeGetDeclaredTypeWithName(str)) {
      return type;
    }
  }
  return nullptr;
}

TypeExpr *Unifier::maybeGetDeclaredTypeWithName(llvm::StringRef name) {
  TRACE();
  DBGS("Getting type: " << name << '\n');
  if (auto *type = typeEnv.lookup(name)) {
    return clone(type);
  }
  DBGS("Type not declared: " << name << '\n');
  return nullptr;
}

TypeExpr *Unifier::getDeclaredType(llvm::StringRef name) {
  TRACE();
  if (auto *type = maybeGetDeclaredType(name)) {
    return type;
  }
  RNULL("Type not declared: " + name.str());
}

TypeExpr *Unifier::getDeclaredType(Node node) {
  TRACE();
  return getDeclaredType(getTextSaved(node));
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
  if (env.count(str)) {
    DBGS("WARNING: Type of " << name << " redeclared\n");
  }
  env.insert(str, type);
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

TypeExpr* Unifier::getVariableType(std::vector<std::string> path) {
  auto hashedPath = hashPath(path);
  return getVariableType(stringArena.save(hashedPath));
}
TypeExpr* Unifier::getVariableType(const std::string_view name) {
  return getVariableType(stringArena.save(name));
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
  if (auto *type = maybeGetVariableTypeWithName(name)) {
    return type;
  }
  for (auto &path : llvm::reverse(moduleSearchPath)) {
    auto possiblePath = hashPath(std::vector<std::string>{path.str(), name.str()});
    auto str = stringArena.save(possiblePath);
    DBGS("Checking module search path: " << path << " with name: " << str << '\n');
    if (auto *type = maybeGetVariableTypeWithName(str)) {
      return type;
    }
  }
  DBGS("Type not declared: " << name << '\n');
  return nullptr;
}

TypeExpr *Unifier::maybeGetVariableTypeWithName(const llvm::StringRef name) {
  DBGS(name << '\n');
  if (auto *type = env.lookup(name)) {
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
  if (currentModule.size() > 0) {
    auto currentModulePath = currentModule;
    currentModulePath.insert(currentModulePath.end(), path.begin(), path.end());
    return hashPath(currentModulePath);
  }
  return hashPath(path);
}

} // namespace ocamlc2
